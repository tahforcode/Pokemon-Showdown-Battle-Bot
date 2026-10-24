"""Gymnasium environment + opponents for self-play on a LOCAL Showdown server."""
import functools
import glob
import os
import random

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from poke_env.environment import SinglesEnv, SingleAgentWrapper
from poke_env.player import (
    MaxBasePowerPlayer,
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)

from .features import embed, embedding_size

FORMAT = "gen9randombattle"


class ShowdownEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(battle_format=FORMAT, strict=False, **kwargs)
        n = embedding_size()
        self.observation_spaces = {
            a: Box(-3.0, 3.0, shape=(n,), dtype=np.float32) for a in self.possible_agents
        }

    def embed_battle(self, battle):
        return embed(battle)

    def calc_reward(self, battle):
        # Small shaped rewards for HP/knockouts, big one for the win.
        # Shaping speeds early learning; the win term is what actually matters.
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=15.0
        ) / 15.0


class PolicyPlayer(Player):
    """Plays using a saved MaskablePPO model. Used for past-self opponents and for laddering."""

    def __init__(self, model, deterministic=False, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.deterministic = deterministic

    def choose_move(self, battle):
        mask = np.array(SinglesEnv.get_action_mask(battle), dtype=bool)
        if not mask.any():
            return self.choose_random_move(battle)
        action, _ = self.model.predict(
            embed(battle), action_masks=mask, deterministic=self.deterministic
        )
        return SinglesEnv.action_to_order(np.int64(action), battle, strict=False)


class PoolOpponent(Player):
    """Picks a new opponent every battle: fixed baselines plus recent snapshots of the agent itself.

    Mixing old selves in prevents the agent from 'forgetting' how to beat strategies
    it used to beat (a classic self-play failure mode).
    """

    def __init__(self, snapshot_dir, self_play_prob=0.5, max_snapshots=10, **kwargs):
        super().__init__(**kwargs)
        kw = dict(battle_format=FORMAT, start_listening=False)
        self.baselines = {
            "random": RandomPlayer(**kw),
            "maxpower": MaxBasePowerPlayer(**kw),
            "heuristic": SimpleHeuristicsPlayer(**kw),
        }
        self.snapshot_dir = snapshot_dir
        self.self_play_prob = self_play_prob
        self.max_snapshots = max_snapshots
        self._loaded = {}
        self.current_name = "random"
        self.current = self.baselines["random"]

    def _snapshots(self):
        files = sorted(glob.glob(os.path.join(self.snapshot_dir, "*.zip")), key=os.path.getmtime)
        return files[-self.max_snapshots:]

    def _pick(self):
        snaps = self._snapshots()
        # Drop opponents that have rotated out of the newest-N window, so memory stays flat.
        for old in [p for p in self._loaded if p not in snaps]:
            del self._loaded[old]
        if snaps and random.random() < self.self_play_prob:
            path = random.choice(snaps)
            if path not in self._loaded:
                from sb3_contrib import MaskablePPO
                self._loaded[path] = PolicyPlayer(
                    MaskablePPO.load(path, device="cpu"), battle_format=FORMAT, start_listening=False
                )
            self.current_name = "self:" + os.path.basename(path)
            self.current = self._loaded[path]
        else:
            # Weight the hard baseline more once there's something to learn from.
            self.current_name = random.choices(
                ["random", "maxpower", "heuristic"], weights=[1, 2, 4]
            )[0]
            self.current = self.baselines[self.current_name]

    def reset_battles(self):
        super().reset_battles()
        self._pick()

    def choose_move(self, battle):
        return self.current.choose_move(battle)


class SB3Wrapper(gym.Wrapper):
    """Flattens poke-env's {observation, action_mask} dict for sb3-contrib's MaskablePPO,
    and reports win/loss + opponent in `info` at episode end."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space["observation"]
        self.action_space = env.action_space
        self._mask = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._mask = obs["action_mask"]
        return obs["observation"], info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self._mask = obs["action_mask"]
        if term or trunc:
            battle = self.env.env.battle1
            info = dict(info, won=bool(battle.won), opponent=self.env.opponent.current_name)
        return obs["observation"], r, term, trunc, info

    def action_masks(self):
        m = self._mask.astype(bool)
        if not m.any():
            m = m.copy()
            m[0] = True
        return m


def build_env(snapshot_dir, self_play_prob=0.5):
    base = ShowdownEnv()
    opp = PoolOpponent(
        snapshot_dir, self_play_prob=self_play_prob, battle_format=FORMAT, start_listening=False
    )
    return SB3Wrapper(SingleAgentWrapper(base, opp))


def make_env(snapshot_dir, self_play_prob=0.5):
    # functools.partial (not a closure) so it pickles for spawn-based subprocesses
    return functools.partial(build_env, snapshot_dir, self_play_prob)
