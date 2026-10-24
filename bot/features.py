"""Turns a poke-env Battle into a fixed-size float vector the network can read.

Kept deliberately hand-crafted and small: type matchups, HP, boosts, move power.
This is the single biggest lever on how fast the agent learns — add features here
(hazards, items, speed comparisons, revealed opponent moves...) as you iterate.
"""
import numpy as np
from poke_env.battle import Battle, MoveCategory, PokemonType, Status, Weather

TYPES = [t for t in PokemonType if t.name not in ("THREE_QUESTION_MARKS", "STELLAR")]
STATUSES = list(Status)
BOOSTS = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
WEATHERS = list(Weather)


def _types(mon):
    v = np.zeros(len(TYPES), np.float32)
    if mon is not None:
        for t in mon.types:
            if t in TYPES:
                v[TYPES.index(t)] = 1
    return v


def _status(mon):
    v = np.zeros(len(STATUSES), np.float32)
    if mon is not None and mon.status is not None:
        v[STATUSES.index(mon.status)] = 1
    return v


def _boosts(mon):
    if mon is None:
        return np.zeros(len(BOOSTS), np.float32)
    return np.array([mon.boosts.get(b, 0) / 6 for b in BOOSTS], np.float32)


def _mon_block(mon):
    hp = mon.current_hp_fraction if mon is not None else 0.0
    spe = (mon.base_stats.get("spe", 0) / 200) if mon is not None else 0.0
    return np.concatenate([[hp, spe], _types(mon), _status(mon), _boosts(mon)])


def _move_block(move, me, opp):
    """7 numbers per move slot: power, effectiveness, STAB, acc, pp, physical/special/status."""
    if move is None:
        return np.zeros(8, np.float32)
    eff = opp.damage_multiplier(move) if opp is not None else 1.0
    stab = 1.0 if (me is not None and move.type in me.types) else 0.0
    cat = move.category
    return np.array([
        move.base_power / 150,
        np.log2(eff + 1e-3) / 2 if eff > 0 else -2.0,  # -2 = immune, 0 = neutral, +1 = 4x
        stab,
        move.accuracy if isinstance(move.accuracy, float) else 1.0,
        (move.current_pp / move.max_pp) if move.max_pp else 0.0,
        float(cat == MoveCategory.PHYSICAL),
        float(cat == MoveCategory.SPECIAL),
        move.priority / 3,
    ], np.float32)


def _matchup(mon, opp):
    """How well `mon`'s STAB types hit `opp`, and how well `opp`'s types hit `mon`."""
    if mon is None or opp is None:
        return [0.0, 0.0]
    off = max((opp.damage_multiplier(t) for t in mon.types if t is not None), default=1)
    dfn = max((mon.damage_multiplier(t) for t in opp.types if t is not None), default=1)
    return [np.log2(off + 1e-3) / 2, np.log2(dfn + 1e-3) / 2]


def embed(battle: Battle) -> np.ndarray:
    me, opp = battle.active_pokemon, battle.opponent_active_pokemon

    moves = list(me.moves.values())[:4] if me is not None else []
    moves += [None] * (4 - len(moves))
    move_feats = np.concatenate([_move_block(m, me, opp) for m in moves])

    team = list(battle.team.values())[:6]
    team += [None] * (6 - len(team))
    bench = np.concatenate([
        np.array([
            m.current_hp_fraction if m else 0.0,
            float(m.fainted) if m else 1.0,
            float(m.active) if m else 0.0,
            *_matchup(m, opp),
        ], np.float32)
        for m in team
    ])

    weather = np.zeros(len(WEATHERS), np.float32)
    for w in battle.weather:
        if w in WEATHERS:
            weather[WEATHERS.index(w)] = 1

    opp_fainted = sum(m.fainted for m in battle.opponent_team.values())
    my_fainted = sum(m.fainted for m in battle.team.values())
    global_feats = np.array([
        my_fainted / 6,
        opp_fainted / 6,
        len(battle.opponent_team) / 6,  # how much of their team we've seen
        float(battle.can_tera),
        min(battle.turn, 100) / 100,
    ], np.float32)

    return np.concatenate([
        _mon_block(me), _mon_block(opp), move_feats, bench, weather, global_feats,
    ]).astype(np.float32).clip(-3, 3)


def embedding_size() -> int:
    return (2 + len(TYPES) + len(STATUSES) + len(BOOSTS)) * 2 + 8 * 4 + 5 * 6 + len(WEATHERS) + 5
