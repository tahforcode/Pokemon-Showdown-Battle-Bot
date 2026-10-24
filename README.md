# Showdown self-play bot

A reinforcement-learning agent for **Gen 9 Random Battles** that learns by playing thousands of games against baseline bots and frozen copies of its past self, on a Showdown server running on your own machine.

## Setup (once)

```bash
# 1. Local Showdown server (needs Node 18+)
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && npm install && cp config/config-example.js config/config.js
node pokemon-showdown start --no-security        # leave running in its own terminal

# 2. Python deps (3.10+)
cd showdown-bot
pip install -r requirements.txt
```

## Train

```bash
python train.py --envs 8          # runs until you Ctrl+C; resumes automatically next time
tensorboard --logdir logs         # optional: live charts
```

Every 50k decisions the current policy is frozen into `snapshots/`; half the games are then played against a random recent snapshot, the rest against three built-in bots (random, always-strongest-move, and a type-aware heuristic). Win rates per opponent print every 2k steps and go to `logs/winrate.csv`. **The number to watch is `heuristic`** — that bot is roughly a low-ladder human.

`--envs` = parallel battles. Set it to about your CPU core count.

## Play it

```bash
python play.py eval --games 100   # benchmark vs the heuristic bot
python play.py accept             # then open https://play.pokemonshowdown.com/~~localhost:8000
                                  # and challenge "ShowdownBot" to a Gen 9 Random Battle
python play.py ladder --games 20 --user YourBotAccount --password '...'
```

Only ladder once it reliably beats the heuristic bot locally, and check Showdown's current rules on bot accounts first.

On Windows, use `py` instead of `python` (e.g. `py train.py --envs 8`), and start the Showdown server before training — the bot can't connect without it.

## Files

- `bot/features.py` — what the agent "sees" (HP, types, boosts, move power and type effectiveness, bench matchups). **The best place to improve it.**
- `bot/env.py` — the environment, reward, past-self opponent pool.
- `train.py` — the training loop. `play.py` — evaluate / play humans / ladder.

## Where to take it next

1. **More features**: entry hazards, items/abilities, speed comparison, opponent's revealed moves, whether you outspeed.
2. **Kaggle/replay data as a warm start**: pretrain the policy to imitate good players' moves (behaviour cloning), then continue with self-play. This usually cuts training time a lot; pure trial-and-error from zero is slow.
3. **Memory**: an LSTM policy (`RecurrentPPO` in sb3-contrib) so it can remember what the opponent revealed earlier.
