import argparse


def make_parser():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ttt", action='store_true', help="Play Tic-Tac-Toe", default=True)
    group.add_argument("--snakes", action='store_true', help="Play Multiplayer Snakes", default=False)
    group.add_argument("--cards", action='store_true', help="Play Cards", default=False)

    parser.add_argument("--lr", type=float, default=1e-4, metavar="F")
    parser.add_argument("--max-steps", type=int, default=50, metavar="N")
    parser.add_argument("--checkpoint", type=str, default=None, metavar="FILE")
    parser.add_argument("--checkpoint-dir", type=str, default="./logs", metavar="FILE")
    return parser


config = dict(
    game="ttt442",
    path=None,
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


def main():
    import azero
    import wandb

    parser = make_parser()
    args = parser.parse_args()
    print(args)
    cfg = dict(**config)
    cfg["path"] = args.checkpoint_dir
    cfg["max_steps"] = args.max_steps
    cfg["lr"] = args.lr

    wandb.init(config=config, project="test", name=None)
    if args.ttt:
        from games.tic_tac_toe import register_pyspiel
        register_pyspiel(4, 4, 2, "ttt442")
        cfg["game"] = "ttt442"
    else:
        assert False, "not yet"

    with azero.spawn.main_handler():
        azero.alpha_zero(azero.Config(**config), is_win_loose=True, checkpoint=args.checkpoint, start_step=1)
    for _ in range(2): # first one symlinks to W&B directory, second saves now
        wandb.save(args.checkpoint_dir + "*")
    wandb.finish()



if __name__ == "__main__":
    main()
