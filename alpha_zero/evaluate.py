"""
Evaluation of a bot in play against multiple MCTS bots.

Use command line arguments to specify which bot to test
and where to save the measured data.

Usage
=====
    see `evaluate.py --help`

"""
import argparse
import itertools
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tf logs and warnings

import pandas as pd
import pyspiel
import tqdm
from azero import load_mcts_bot, load_trained_bot

import wandb
from games import SNAKES_NAME, TTT_NAME

MCTS_SIMULS = [0, 5, 10, 15, 20, 50, 120, 250, 500]
RANDOM_PLAYER = lambda state: random.choice(state.legal_actions())



################################################################################
##                            HELPERS
################################################################################

def _restore_checkpoint_files(path: str, chkt: int, run_path: str, move_path: str):
    chkt_path = os.path.join(path, f"checkpoint-{chkt}")

    if os.path.exists(move_path):
        logging.warn(f"Destination folder {move_path=} exists, skipping...")
        return

    wandb.restore(os.path.join(path, "config.json"), run_path=run_path)
    for suffix in [".index", ".meta", ".data-00000-of-00001"]:
        wandb.restore(chkt_path + suffix, run_path=run_path)

    dir_path = os.path.dirname(chkt_path)
    logging.info(f"Moving loaded files from {dir_path=} to {move_path=}")
    shutil.move(dir_path, move_path)


"""(Unused) config to initialise a MCTS agent (from azero)."""
_mcts_unused_names = [
    'game', 'path', 'learning_rate', 'weight_decay', 'train_batch_size',
    'replay_buffer_size', 'replay_buffer_reuse', 'max_steps', 'checkpoint_freq',
    'actors', 'evaluators', 'evaluation_window', 'eval_levels', 'policy_alpha',
    'policy_epsilon', 'temperature', 'temperature_drop', 'nn_model',
    'nn_width', 'nn_depth', 'observation_shape',   'output_size', 'quiet'
]
_MCTS_UNUSED_CFG = { name: None for name in _mcts_unused_names }


@dataclass
class Result:
    player: str
    mcts_simuls: int
    mcts_rate: float
    player_first: bool
    result_from_player: int
    score_diff_from_player: int
    moves: list


################################################################################
##                            PARSER
################################################################################

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--path", type=str, metavar="PATH", help="Where to save generated .csv (e.g. `results.csv`)")

    game = parser.add_mutually_exclusive_group()
    game.add_argument("--ttt", action='store_true', help="Tic-Tac-Toe", default=False)
    game.add_argument("--snakes", action='store_true', help="Multiplayer Snakes", default=False)
    game.add_argument("--cards", action='store_true', help="Cards", default=False)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--random", action='store_true', help="Evaluate Random Player", default=False)
    group.add_argument("--trained", action='store_true', help="Evaluate Trained Agent", default=False)

    parser.add_argument("--runname", type=str, default=None, metavar="NAME", help="Name of the run")
    parser.add_argument("--logs", type=str, default="./logs", metavar="DIR", help="Directory with stored checkpoints on w&b")
    parser.add_argument("--id", type=str, default=None, metavar="PATH", help="w&b id of the run")
    parser.add_argument("--checkpoint", type=int, default=-1, metavar="N", help="which checkpoint")

    parser.add_argument("--games", type=int, default=20, metavar="N", help="Number of games")
    parser.add_argument("--mcts-rate", type=float, default=1.4, metavar="F", help="MCTS exploration constant")

    return parser


################################################################################
##                            MAIN
################################################################################

def load_game(args):
    if args.ttt:
        name = TTT_NAME
    elif args.snakes:
        name = SNAKES_NAME
    elif args.cards:
        name = "nim"
    else:
        raise ValueError("Must provide game to evaluate on.")

    return pyspiel.load_game(name), name


def load_player_fn(args):
    if args.random:
        if args.runname is None:
            args.runname = "random"
        return RANDOM_PLAYER
    else:
        assert args.runname is not None
        assert args.logs is not None
        assert args.id is not None
        _restore_checkpoint_files(args.logs, args.checkpoint, args.id, args.runname)

        with open(os.path.join(args.runname, "config.json"), "r") as f:
            cfg = json.load(f)
            print("Loaded config:", cfg)
            # assume correct game name there

        bot, _ = load_trained_bot(cfg, args.runname, args.checkpoint, is_eval=True)
        return bot.step


def main(arguments=None, namespace=None):
    parser = make_parser()
    args = parser.parse_args(args=arguments, namespace=namespace)

    random.seed(0)
    results: list[Result] = []
    game, game_name = load_game(args)
    player_fn = load_player_fn(args)

    for mcts_simuls in tqdm.tqdm(MCTS_SIMULS, desc="mcts", leave=None):
        if mcts_simuls == 0:
            mcts_fn = RANDOM_PLAYER
        else:
            mcts_cfg = dict(game=game_name, uct_c=args.mcts_rate, max_simulations=mcts_simuls)
            mcts_bot = load_mcts_bot(_MCTS_UNUSED_CFG | mcts_cfg, is_eval=True)
            mcts_fn = mcts_bot.step

        for i in tqdm.trange(args.games, leave=None, desc=f"{args.runname} vs. mcts({mcts_simuls})"):
            state = game.new_initial_state()
            players = [player_fn, mcts_fn] if i % 2 == 0 else [mcts_fn, player_fn]
            actions = []

            for p in itertools.cycle(players):
                if state.is_terminal():
                    break
                action = p(state)
                state.apply_action(action)
                actions.append(action)

            player_res = state.returns()[i % 2]
            try:
                p1, p2 = state._game._scores
            except AttributeError:
                p1 = p2 = 0
            score_diff = p1 - p2 if i % 2 == 0 else p2 - p1
            res = Result(args.runname, mcts_simuls, args.mcts_rate, i % 2 == 0, player_res, score_diff, actions)
            results.append(res)

    df = pd.DataFrame([vars(x) for x in results])
    print(df)
    df.to_csv(args.path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.NOTSET, format='[%(asctime)s] %(levelname)s: %(message)s')
    main()
