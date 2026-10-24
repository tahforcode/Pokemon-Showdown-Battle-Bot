"""Put a trained model in front of real opponents.

Local server (play against it yourself in the browser):
    python play.py accept
    -> open https://play.pokemonshowdown.com/~~localhost:8000 and challenge "ShowdownBot"
       to a Gen 9 Random Battle.

Public ladder (needs a registered Showdown account; keep game counts modest and
check the site's current rules on bots first):
    python play.py ladder --games 20 --user YourBotName --password '...'

Quick benchmark vs the built-in heuristic bot (local server):
    python play.py eval --games 100
"""
import argparse
import asyncio
import os

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import SimpleHeuristicsPlayer
from sb3_contrib import MaskablePPO

from bot.env import FORMAT, PolicyPlayer

ROOT = os.path.dirname(os.path.abspath(__file__))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["accept", "ladder", "eval"])
    ap.add_argument("--model", default=os.path.join(ROOT, "checkpoints", "latest.zip"))
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--user", default="ShowdownBot")
    ap.add_argument("--password", default=None)
    args = ap.parse_args()

    model = MaskablePPO.load(args.model, device="cpu")

    if args.mode == "ladder":
        if not args.password:
            raise SystemExit("--password is required for the public server")
        bot = PolicyPlayer(
            model,
            deterministic=True,
            battle_format=FORMAT,
            account_configuration=AccountConfiguration(args.user, args.password),
            server_configuration=ShowdownServerConfiguration,
            start_timer_on_battle_start=True,
        )
        await bot.ladder(args.games)
    elif args.mode == "accept":
        bot = PolicyPlayer(
            model,
            deterministic=True,
            battle_format=FORMAT,
            account_configuration=AccountConfiguration(args.user, None),
        )
        print(f"Waiting for challenges to {args.user} on localhost:8000 ...")
        await bot.accept_challenges(None, args.games)
    else:
        bot = PolicyPlayer(model, deterministic=True, battle_format=FORMAT)
        opp = SimpleHeuristicsPlayer(battle_format=FORMAT)
        await bot.battle_against(opp, n_battles=args.games)

    print(f"Record: {bot.n_won_battles}W / {bot.n_lost_battles}L of {bot.n_finished_battles}")


if __name__ == "__main__":
    asyncio.run(main())
