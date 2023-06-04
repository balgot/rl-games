from open_spiel.python.algorithms import dqn
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import random_agent

from games.tic_tac_toe import register_pyspiel

import numpy as np
import tensorflow.compat.v1 as tf


def eval_agents(env, opponent_pairs, num_episodes):
    num_players = len(opponent_pairs[0])
    sum_episode_rewards = np.zeros((len(opponent_pairs), num_players))
    for pair_idx, opponents in enumerate(opponent_pairs):
        for _ in range(num_episodes):
            time_step = env.reset()
            episode_rewards = np.zeros(num_players)
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if env.is_turn_based:
                    agent_output = opponents[player_id].step(
                        time_step, is_evaluation=True)
                    action_list = [agent_output.action]
                else:
                    agents_output = [
                        agent.step(time_step, is_evaluation=True) for agent in opponents
                    ]
                    action_list = [agent_output.action for agent_output in agents_output]

                time_step = env.step(action_list)
                episode_rewards += time_step.rewards
            sum_episode_rewards[pair_idx] += episode_rewards

    return sum_episode_rewards / num_episodes


def create_opponent_pairs(agents, random_bots, num_players, num_configs):
    opponent_pairs = []
    # Each agent against random bot
    for ag in agents:
        if ag.player_id == 0:
            opponent_pairs.append((ag, random_bots[1]))
        else:
            opponent_pairs.append((random_bots[0], ag))

    # Same config agents against each other
    for i in range(0, len(agents), num_players):
        opponent_pairs.append((agents[i], agents[i + 1]))

    # Different config agents against each other
    for i in range(num_configs - 1):
        for j in range(i + 1, num_configs):
            opponent_pairs.append((agents[i * num_players], agents[j * num_players + 1]))
            opponent_pairs.append((agents[j * num_players], agents[i * num_players + 1]))

    return opponent_pairs


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
        evaluation = []

        for ep in range(game_config['num_train_episodes']):
            # Evaluate gradually
            if (ep + 1) % game_config['eval_every'] == 0:
                print(f"Evaluating after {ep + 1} episodes.")
                mean_rewards = eval_agents(env, opponent_pairs, game_config['num_eval_episodes'])
                evaluation.append(mean_rewards)

            # Train DQN agents for each config
            for cfg_idx in range(1, num_configs + 1):
                begin, end = (cfg_idx - 1) * num_players, (cfg_idx - 1) * num_players + 1
                curr_agents = agents[begin:end+1]

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

                # Episode is over, step all agents with final info state.
                for agent in curr_agents:
                    agent.step(time_step)
        return evaluation


if __name__ == "__main__":
    register_pyspiel(5, 5, 3, "ttt")

    game_cfg = {
        'game': 'ttt',
        'num_players': 2,
        'num_train_episodes': 10_000,
        'eval_every': 1000,
        'num_eval_episodes': 250
    }

    TTT_cfg_small = {
        'name': 'TTT_small',
        'batch_size': 32,
        'learn_every': 2,
        'update_target_network_every': 100,
        'optimizer': 'adam',
        'hidden_layers_sizes': [32, 32]
    }

    TTT_cfg_big = {
        'name': 'TTT_big',
        'batch_size': 32,
        'learn_every': 2,
        'update_target_network_every': 100,
        'optimizer': 'adam',
        'hidden_layers_sizes': [64, 128, 64]
    }

    rewards = train_eval(game_cfg, [TTT_cfg_small, TTT_cfg_big])
