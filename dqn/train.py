from open_spiel.python.algorithms import dqn
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

from games.tic_tac_toe import TTT_NAME
from games.snakes import SNAKES_NAME

import tensorflow.compat.v1 as tf

from evaluate import eval_agents, create_opponent_pairs


def train_eval(game_config, agents_config):
    env = rl_environment.Environment(game_config['game'])
    info_state_size = env.observation_spec()['info_state'][0]
    num_actions = env.action_spec()['num_actions']
    num_players = game_config['num_players']

    random_bots = [random_agent.RandomAgent(player_id=idx, num_actions=num_actions) for idx in range(num_players)]

    with tf.Session() as sess:
        # set up num_players of agents for each DQN configuration
        agents = []
        for cfg in agents_config:
            agents.extend([
                dqn.DQN(
                    session=sess,
                    player_id=idx,
                    state_representation_size=info_state_size,
                    num_actions=num_actions,
                    hidden_layers_sizes=cfg['hidden_layers_sizes'],
                    batch_size=cfg['batch_size'],
                    optimizer_str=cfg['optimizer'],
                    learn_every=cfg['learn_every'],
                    update_target_network_every=cfg['update_target_network_every']) for idx in range(num_players)
            ])
        sess.run(tf.global_variables_initializer())

        num_configs = len(agents) // num_players
        opponent_pairs = create_opponent_pairs(agents, random_bots, num_players, num_configs)

        rewards = []
        losses = []
        for ep in range(game_config['num_train_episodes']):
            # Evaluate gradually
            if (ep + 1) % game_config['eval_every'] == 0:
                print(f"Evaluating after {ep + 1} episodes.")
                mean_rewards = eval_agents(env, opponent_pairs, game_config['num_eval_episodes'])
                rewards.append(mean_rewards)

            # Train DQN agents for each config
            for cfg_idx in range(1, num_configs + 1):
                # Grab agents for current config
                begin, end = (cfg_idx - 1) * num_players, (cfg_idx - 1) * num_players + 1
                curr_agents = agents[begin:end + 1]

                # Simulate a full game
                time_step = env.reset()
                while not time_step.last():
                    player_id = time_step.observations["current_player"]
                    if env.is_turn_based:
                        agent_output = curr_agents[player_id].step(time_step)
                        action_list = [agent_output.action]
                    else:
                        agents_output = [agent.step(time_step) for agent in curr_agents]
                        action_list = [agent_output.action for agent_output in agents_output]
                    time_step = env.step(action_list)

                # Step all agents with final info state
                for agent in curr_agents:
                    agent.step(time_step)

        return losses, rewards


if __name__ == "__main__":
    # Train three DQN with increasing size
    DQN_small = {
        'name': 'DQN_small',
        'batch_size': 32,
        'learn_every': 2,
        'update_target_network_every': 100,
        'optimizer': 'adam',
        'hidden_layers_sizes': [32, 32]
    }

    DQN_medium = {
        'name': 'DQN_medium',
        'batch_size': 64,
        'learn_every': 2,
        'update_target_network_every': 100,
        'optimizer': 'adam',
        'hidden_layers_sizes': [256, 128, 256]
    }

    DQN_large = {
        'name': 'DQN_large',
        'batch_size': 128,
        'learn_every': 2,
        'update_target_network_every': 100,
        'optimizer': 'adam',
        'hidden_layers_sizes': [1024, 512, 512, 512, 1024]
    }

    DQN_configs = [DQN_small, DQN_medium, DQN_large]

    # Train on TTT, Snakes and Nim
    game_cfg = {
        'game': TTT_NAME,
        'num_players': 2,
        'num_train_episodes': 10_000,
        'eval_every': 1000,
        'num_eval_episodes': 250
    }

    # TTT
    TTT_losses, TTT_gradual_rewards = train_eval(game_cfg, DQN_configs)

    # # Snakes
    # game_cfg['game'] = SNAKES_NAME
    # Snakes_losses, Snakes_gradual_rewards = train_eval(game_cfg, DQN_configs)
    #
    # # Nim
    # game_cfg['game'] = 'nim'
    # Nim_losses, Nim_gradual_rewards = train_eval(game_cfg, DQN_configs)
