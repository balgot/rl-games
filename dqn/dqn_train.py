import tensorflow.compat.v1 as tf
from open_spiel.python.algorithms import dqn
from open_spiel.python import rl_environment

from games.tic_tac_toe import register_pyspiel


def train(config):
    env = rl_environment.Environment(config['game'])
    info_state_size = env.observation_spec()['info_state'][0]
    num_actions = env.action_spec()['num_actions']
    num_players = config['num_of_players']

    with tf.Session() as sess:
        hidden_layers_sizes = [int(layer_size) for layer_size in config['hidden_layers_sizes']]
        # pylint: disable=g-complex-comprehension
        agents = [
            dqn.DQN(
                session=sess,
                player_id=idx,
                state_representation_size=info_state_size,
                num_actions=num_actions,
                hidden_layers_sizes=hidden_layers_sizes,
                replay_buffer_capacity=config['replay_buffer_capacity'],
                batch_size=config['batch_size']) for idx in range(num_players)
        ]
        sess.run(tf.global_variables_initializer())

        for ep in range(config['num_train_episodes']):
            if (ep + 1) % config['eval_every'] == 0:
                # TODO implement evaluation step if needed
                pass
            if (ep + 1) % config['save_every'] == 0:
                # FIXME seems like saving checkpoints is not working properly
                for agent in agents:
                    agent.save(config['checkpoint_dir'])

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if env.is_turn_based:
                    agent_output = agents[player_id].step(time_step)
                    action_list = [agent_output.action]
                else:
                    agents_output = [agent.step(time_step) for agent in agents]
                    action_list = [agent_output.action for agent_output in agents_output]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)


if __name__ == "__main__":
    register_pyspiel(4, 4, 2, "ttt442")

    TTT_cfg = {
        'game': 'ttt442',
        'num_of_players': 2,
        'hidden_layers_sizes': [64, 64],
        'replay_buffer_capacity': int(1e5),
        'batch_size': 32,
        'num_train_episodes': int(10),
        'eval_every': int(1e3),
        'save_every': int(1e3),
        'checkpoint_dir': '/tmp/dqn_test'
    }

    train(TTT_cfg)
