import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tf logs and warnings


def make_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ttt", action='store_true', help="Play Tic-Tac-Toe", default=False)
    group.add_argument("--snakes", action='store_true', help="Play Multiplayer Snakes", default=False)
    group.add_argument("--cards", action='store_true', help="Play Cards", default=False)

    parser.add_argument("--lr", type=float, default=1e-4, metavar="F", help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=50, metavar="N", help="Number of epochs")
    parser.add_argument("--uct-c", type=float, default=1.4, metavar="F", help="Exploration constant of MCTS")
    parser.add_argument("--max-simulations", type=int, default=50, metavar="N", help="Max number of MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=128, metavar="N", help="Batch size of one epoch")
    parser.add_argument("--replay-buffer-size", type=int, default=2**14, metavar="N", help="Max size of experience replay buffer")

    parser.add_argument("--checkpoint", type=str, default=None, metavar="FILE", help="Checkpoint from which to continue training")
    parser.add_argument("--checkpoint-freq", type=int, default=10, metavar="N", help="How often save checkpoints")
    parser.add_argument("--checkpoint-dir", type=str, default="./logs", metavar="FILE", help="Folder to save logs and checkpoints to")

    parser.add_argument("--model", type=str, choices=["mlp", "conv2d", "resnet"], default="mlp", help="Model type")
    parser.add_argument("--nn-width", type=int, default=256, metavar="N", help="Hidden layer size")
    parser.add_argument("--nn-depth", type=int, default=3, metavar="N", help="Number of hidden layers in torso")

    parser.add_argument("--wandbproject", type=str, default="test", metavar="STR", help="wandb project name")
    parser.add_argument("--wandbname", type=str, default=None, metavar="STR", help="Run name")
    return parser


config = dict(
    weight_decay=1e-4,
    replay_buffer_reuse=4,
    actors=4,
    evaluators=2,
    policy_alpha=0.25,
    policy_epsilon=1,
    temperature=1,
    temperature_drop=4,
    evaluation_window=50,
    eval_levels=7,
    observation_shape=None,
    output_size=None,
    quiet=True,
)


def main():
    import azero
    import wandb

    parser = make_parser()
    args = parser.parse_args()
    cfg = dict(**config)

    cfg["path"] = args.checkpoint_dir
    cfg["max_steps"] = args.max_steps
    cfg["learning_rate"] = args.lr
    cfg["train_batch_size"] = args.batch_size
    cfg["replay_buffer_size"] = args.replay_buffer_size
    cfg["checkpoint_freq"] = args.checkpoint_freq
    cfg["max_simulations"] = args.max_simulations
    cfg["nn_model"] = args.model
    cfg["nn_width"] = args.nn_width
    cfg["nn_depth"] = args.nn_depth
    cfg["uct_c"] = args.uct_c

    wandb.init(config=config, project=args.wandbproject, name=args.wandbname)
    if args.ttt:
        from games import TTT_NAME  # it registers the game
        cfg["game"] = TTT_NAME
    elif args.snakes:
        from games import SNAKES_NAME  # also registers the game
        cfg["game"] = SNAKES_NAME
    elif args.cards:
        cfg["game"] = "nim"
    else:
        raise ValueError("Must provide which game to train on")

    print(f"\n\nconfig={cfg}\n\n")
    with azero.spawn.main_handler():
        azero.alpha_zero(azero.Config(**cfg), is_win_loose=True, checkpoint=args.checkpoint)

    for _ in range(2): # first one symlinks to W&B directory, second saves now
        wandb.save(os.path.join(args.checkpoint_dir, "*"))
    wandb.finish()


if __name__ == "__main__":
    main()
