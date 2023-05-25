import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../games"))

import azero
import wandb
from tic_tac_toe import register_pyspiel
register_pyspiel(4, 4, 2, "ttt442")


# for now, only TTT


if __name__ == "__main__":
    checkpoint = None
    checkpoint_dir = "logs/"

    config = dict(
        game="ttt442",
        path=checkpoint_dir,
        learning_rate=0.0001,
        weight_decay=1e-4,
        train_batch_size=256,
        replay_buffer_size=2**14,
        replay_buffer_reuse=8,
        max_steps=10,
        checkpoint_freq=50,

        actors=4,
        evaluators=2,
        uct_c=1.4,
        max_simulations=50,
        policy_alpha=0.25,
        policy_epsilon=1,
        temperature=1,
        temperature_drop=4,
        evaluation_window=50,
        eval_levels=7,

        nn_model="mlp",
        nn_width=256,
        nn_depth=2,
        observation_shape=None,
        output_size=None,

        quiet=True,
    )

    wandb.init(config=config, project="test", name=None)
    with azero.spawn.main_handler():
        azero.alpha_zero(azero.Config(**config), is_win_loose=True, checkpoint=checkpoint, start_step=1)
    for _ in range(2): # first one symlinks to W&B directory, second saves now
        wandb.save(checkpoint_dir + "*")
    wandb.finish()
