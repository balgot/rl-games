import numpy as np


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


def eval_agents(env, opponent_pairs, num_episodes):
    sum_episode_rewards = np.zeros(len(opponent_pairs))
    # Evaluate each agent pair
    for pair_idx, opponents in enumerate(opponent_pairs):
        for _ in range(num_episodes):
            # Simulate full game
            time_step = env.reset()
            episode_rewards = 0
            while not time_step.last():
                player_id = time_step.observations["current_player"]

                agent_output = opponents[player_id].step(time_step, is_evaluation=True)
                action_list = [agent_output.action]

                time_step = env.step(action_list)
                # Zero sum game
                episode_rewards += time_step.rewards[0]

            # Accumulate rewards
            sum_episode_rewards[pair_idx] += episode_rewards

    return sum_episode_rewards / num_episodes
