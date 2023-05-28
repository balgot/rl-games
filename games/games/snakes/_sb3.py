import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
import numpy as np
from ._game import Snakes, PLAYER1, PLAYER2, _PLAYERS, FRUIT

assert False, "WIP"

class SnakeEnv(gym.Env):
    metadata = { "render_modes": [None] }
    ACTIONS = ["WASD"]

    def __init__(self, player, rows, cols, render_mode=None):
        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(low=-2, high=3, shape=(rows, cols), dtype=int)

        self._player = player
        self.shape = (rows, cols)
        self.render_mode = render_mode
        self.game = Snakes(rows, cols)
        self._next_player = 0

    def _get_obs(self, from_player=0):
        if from_player == 0:
            return self.game.board
        obs = -self.game.board
        obs[obs == -FRUIT] = FRUIT
        return obs

    def _get_info(self):
        return {
            "len1": len(self.game.snakes[PLAYER1]),
            "len2": len(self.game.snakes[PLAYER2]),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._next_player = 1 - self._next_player
        self.game = Snakes(*self.shape)
        observation = self._get_obs(self._next_player)
        info = self._get_info()
        return observation, info

    def step(self, action):
        me = self._next_player
        opp = 1 - me

        assert action in range(len(self.ACTIONS))
        other_action, _ = self._player.predict(self._get_obs(opp))

        self.game.make_move(_PLAYERS[me], action)
        self.game.make_move(_PLAYERS[opp], other_action)

        obs = self._get_obs()
        info = self._get_info()
        trunc = False
        reward = 1
        done = False

        if self.game.is_game_over():
            done = True
            if self.game.winner() == _PLAYERS[me]:
                reward += 100
            elif self.game.winner() is None:
                reward += 10
            else:
                reward += -150

        return obs, reward, done, trunc, info

    def render(self):
        pass

    def close(self):
        pass


class Dummy:
    def predict(self, *args, **kwargs):
        return "W", None

env = SnakeEnv(Dummy(), 10, 10)
check_env(env)