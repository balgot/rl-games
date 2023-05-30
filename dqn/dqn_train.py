import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../games"))

from pyspiel import load_game
from open_spiel.python.algorithms import dqn
from tic_tac_toe import register_pyspiel
register_pyspiel(4, 4, 2, "ttt442")


def train(config):
    game = load_game(config['game'])
    info_state_size = game.observation_tensor_shape()
    num_actions = game.num_distinct_actions()
    num_players = 2

    hidden_layers_sizes = [int(l_size) for l_size in config['hidden_layers_sizes']]

    agents = [
        dqn.DQN(
            session=None,
            player_id=idx,
            state_representation_size=info_state_size,
            num_actions=num_actions,
            hidden_layers_sizes=hidden_layers_sizes,
            replay_buffer_capacity=config['replay_buffer_capacity'],
            batch_size=config['batch_size']) for idx in range(num_players)
    ]

    for ep in range(config['num_train_episodes']):
        if (ep + 1) % config['save_every'] == 0:
            for agent in agents:
                agent.save(config['checkpoint_dir'])

        time_step = game.reset()
        while not time_step.last():
            player_id = time_step.observations["current_player"]

            agent_output = agents[player_id].step(time_step)
            action_list = [agent_output.action]

            time_step = game.step(action_list)

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)


if __name__ == "__main__":
    cfg = {
        'game': 'ttt442',
        'hidden_layers_sizes': [64, 64],
        'replay_buffer_capacity': int(1e5),
        'batch_size': 32,
        'num_train_episodes': int(1e6),
        'eval_every': int(1e4),
        'save_every': int(1e4),
        'checkpoint_dir': '/tmp/dqn_test'
    }

    train(cfg)
