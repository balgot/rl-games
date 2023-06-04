import numpy as np
import pyspiel
from ._game import Snakes, ACTIONS, PLAYER1, PLAYER2, PLAYER1_HEAD, PLAYER2_HEAD, FRUIT


_ACTION_NUMS = { i: a for i, a in enumerate(ACTIONS.keys()) }
_MAX_MOVES = 100

# TODO: ignoring randomness of fruit spawn

def register_pyspiel(width: int, height: int, name: str):
    """
    Register Snakes* as a pyspiel game.

    After this, it is possible to use Snakes game within open_spiel
    library for different RL algorithms.

    Arguments
    =========
        width: width of the game plan
        height: height of the game plan
        name: name of the pyspiel game

    Returns
    =======
        nothing
    """
    _GAME_TYPE = pyspiel.GameType(
        short_name=name,
        long_name=name,

        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.SAMPLED_STOCHASTIC,
        information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
        utility=pyspiel.GameType.Utility.ZERO_SUM,
        reward_model=pyspiel.GameType.RewardModel.TERMINAL,

        max_num_players=2,
        min_num_players=2,

        provides_information_state_string=True,
        provides_information_state_tensor=False,
        provides_observation_string=True,
        provides_observation_tensor=True,
        parameter_specification={}
    )

    _GAME_INFO = pyspiel.GameInfo(
        num_distinct_actions=len(ACTIONS),
        max_chance_outcomes=0,
        num_players=2,
        min_utility=-1.0,
        max_utility=1.0,
        utility_sum=0.0,
        max_game_length=min(width * height, _MAX_MOVES)
    )

    class _SnakeGame(pyspiel.Game):
        def __init__(self, params=None):
            super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

        def new_initial_state(self):
            """Returns a state corresponding to the start of a game."""
            return _SnakeState(self)

        def make_py_observer(self, iig_obs_type=None, params=None):
            """Returns an object used for observing game state."""
            _iig = iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False)
            return _SnakeObserver(_iig, params)

    class _SnakeState(pyspiel.State):
        def __init__(self, game):
            super().__init__(game)
            self._game_over = False
            self._game = Snakes(width, height)
            self.player = 0
            self._player0_action = None
            self._move_num = 0

        def current_player(self):
            if self._game_over:
                return pyspiel.PlayerId.TERMINAL
            return self.player

        def _legal_actions(self, player):
            del player
            return list(_ACTION_NUMS.keys())

        def _apply_action(self, action):
            action = _ACTION_NUMS[action]
            if self.player == 0:
                assert self._player0_action is None
                self._player0_action = action
                self.player = 1
            else:
                assert self._player0_action is not None
                self._move_num += 1
                self._game.make_move(PLAYER1, self._player0_action)
                self._game.make_move(PLAYER2, action)
                self._game.step()

        def is_terminal(self):
            return self._game.is_game_over() or self._move_num >= _MAX_MOVES

        def returns(self):
            if not self._game.is_game_over():
                p1 = 0
            else:
                winner = self._game.winner()
                p1 = 0 if winner is None else 1 if winner == PLAYER1 else -1
            return [p1, -p1]

        def _action_to_string(self, player, action):
            return f"{player}:{_ACTION_NUMS[action]}"

        def __str__(self):
            return str(self._game)

    class _SnakeObserver:
        def __init__(self, iig_obs_type, params):
            """Initializes an empty observation tensor."""
            if params:
                raise ValueError(f"Observation parameters not supported; passed {params}")
            # dimensions:
            #   0 - (current) player body
            #   1 - (opposite) player body
            #   2 - (current) player head
            #   3 - (opposite) player head
            #   4 - fruit
            shape = (5, width, height)
            self.tensor = np.zeros(np.prod(shape), np.float32)
            self.dict = { "observation": np.reshape(self.tensor, shape) }

        def set_from(self, state: _SnakeState, player):
            """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
            obs = self.dict["observation"]
            obs.fill(0)

            _heads = [PLAYER1_HEAD, PLAYER2_HEAD]
            _bodys = [PLAYER1, PLAYER2]
            if state.player != 0:
                _heads = _heads[::-1]
                _bodys = _bodys[::-1]
            for idx, value in enumerate([*_bodys, *_heads, FRUIT]):
                obs[idx] = state._game.board == value

        def string_from(self, state, player):
            """Observation of `state` from the PoV of `player`, as a string."""
            return str(state)

    pyspiel.register_game(_GAME_TYPE, _SnakeGame)


if __name__ == "__main__":
    register_pyspiel(5, 5, "snakes")
    game = pyspiel.load_game("snakes")

    state = game.new_initial_state()
    while not state.is_terminal():
        print(state.legal_actions())
        action = np.random.choice(state.legal_actions())
        print(state)
        print(f"{action=}")
        state.apply_action(action)
    print(state)
    print(state.returns())
