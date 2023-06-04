"""Evaluation of trained bot in play against: random, mcts and self-better."""
import itertools
import logging
import os
import random
import shutil
from dataclasses import dataclass
import json

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tf logs and warnings

import pandas as pd
import pyspiel
import tqdm
from azero import load_mcts_bot, load_trained_bot

from games import TTT_NAME
import wandb


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


_MCTS_UNUSED_CFG = {
    'game': None, 'path': None, 'learning_rate': None, 'weight_decay': None,
    'train_batch_size': None, 'replay_buffer_size': None,
    'replay_buffer_reuse': None, 'max_steps': None, 'checkpoint_freq': None,
    'actors': None, 'evaluators': None, 'evaluation_window': None,
    'eval_levels': None, 'policy_alpha': None, 'policy_epsilon': None,
    'temperature': None, 'temperature_drop': None, 'nn_model': None,
    'nn_width': None, 'nn_depth': None, 'observation_shape': None,
    'output_size': None, 'quiet': None
}


RUNS = {
    "random": None,
    "test": ("logs/", "miba/test/18s3o9t6"),
}

MCTS_SIMULS = [0, 5, 10, 15, 20, 50, 120, 250]
MCTS_RATE = 1.4
GAMES = 20


@dataclass
class Result:
    player: str
    mcts_simuls: int
    mcts_rate: float
    player_first: bool
    result_from_player: int
    score_diff_from_player: int
    moves: list


def _eval():
    random.seed(0)
    results = []
    game = pyspiel.load_game(TTT_NAME)

    for player_str in tqdm.tqdm(RUNS.keys(), desc="players"):
        if player_str == "random":
            play_fn = lambda state: random.choice(state.legal_actions())
        else:
            path, run_path = RUNS[player_str]
            _restore_checkpoint_files(path, -1, run_path, player_str)
            with open(os.path.join(player_str, "config.json"), "r") as f:
                cfg = json.load(f)
            bot, _ = load_trained_bot(cfg, player_str, -1, is_eval=True)
            play_fn = bot.step

        for mcts_simuls in tqdm.tqdm(MCTS_SIMULS, desc="mcts", leave=None):
            if mcts_simuls == 0:
                mcts_fn = lambda state: random.choice(state.legal_actions())
            else:
                mcts_cfg = dict(game=TTT_NAME, uct_c=MCTS_RATE, max_simulations=mcts_simuls)
                mcts_bot = load_mcts_bot(_MCTS_UNUSED_CFG | mcts_cfg, is_eval=True)
                mcts_fn = mcts_bot.step

            for i in tqdm.trange(GAMES, leave=None, desc=f"{player_str} vs. mcts({mcts_simuls})"):
                state = game.new_initial_state()
                players = [play_fn, mcts_fn] if i % 2 == 0 else [mcts_fn, play_fn]
                actions = []

                for p in itertools.cycle(players):
                    if state.is_terminal():
                        break
                    action = p(state)
                    state.apply_action(action)
                    actions.append(action)

                player_res = state.returns()[i % 2]
                p1, p2 = state._game._scores
                score_diff = p1 - p2 if i % 2 == 0 else p2 - p1
                res = Result(player_str, mcts_simuls, MCTS_RATE, i % 2 == 0, player_res, score_diff, actions)
                results.append(res)

    df = pd.DataFrame([vars(x) for x in results])
    print(df)
    df.to_csv("evaluation.csv")




if __name__ == "__main__":
    import wandb
    logging.basicConfig(level=logging.NOTSET, format='[%(asctime)s] %(levelname)s: %(message)s')
    _eval()
