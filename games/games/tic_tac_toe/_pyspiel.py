import numpy as np
import pyspiel
from ._game import PLAYERS_STR, TTT, EMPTY


def register_pyspiel(rows: int, cols: int, to_connect: int, name: str):
    """
    Register Tic-Tac-Toe* as a pyspiel game.

    After this, it is possible to use Tic-Tac-Toe within open_spiel
    library for different RL algorithms.

    Arguments
    =========
        rows: number of rows to play on
        cols: number of cols to play on
        to_connect: how many of the same symbols should be connected
        name: name of the pyspiel game

    Returns
    =======
        nothing

    Examples
    ========
        >>> import pyspiel
        >>> register_pyspiel(3, 3, 2, "my_ttt")
        >>> game = pyspiel.load_game("my_tttt")
    """
    _GAME_TYPE = pyspiel.GameType(
        short_name=name,
        long_name=name,

        dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
        chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
        information=pyspiel.GameType.Information.PERFECT_INFORMATION,
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
        num_distinct_actions=rows * cols,
        max_chance_outcomes=0,
        num_players=2,
        min_utility=-1.0,
        max_utility=1.0,
        utility_sum=0.0,
        max_game_length=rows * cols
    )

    class _TTTGame(pyspiel.Game):
        def __init__(self, params=None):
            super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

        def new_initial_state(self):
            """Returns a state corresponding to the start of a game."""
            return _TTTState(self)

        def make_py_observer(self, iig_obs_type=None, params=None):
            """Returns an object used for observing game state."""
            _iig = iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False)
            return TTTObserver(_iig, params)

    class _TTTState(pyspiel.State):
        def __init__(self, game):
            super().__init__(game)
            self._game_over = False
            self._game = TTT(rows, cols, to_connect)

        def current_player(self):
            if self._game_over:
                return pyspiel.PlayerId.TERMINAL
            return self._game._next_player

        def pos2action(self, pos):
            row, col = pos
            return self._game._cols * row + col

        def action2pos(self, action):
            return divmod(action, self._game._cols)

        def _legal_actions(self, player):
            return [self.pos2action(p) for p in self._game.legal_actions()]

        def _apply_action(self, action):
            pos = self.action2pos(action)
            self._game.apply_action(pos)
            self._game_over = self._game.is_full()

        def is_terminal(self):
            return self._game_over

        def returns(self):
            scores = self._game.returns()
            if not self._game_over or scores[0] == scores[1]:
                return [0, 0]
            p1 = 1 if scores[0] > scores[1] else -1
            return [p1, -p1]

        def _action_to_string(self, player, action):
            return f"{PLAYERS_STR[player]}{self.action2pos(action)}"

        @property
        def board(self):
            return self._game.board

        def __str__(self):
            return str(self._game)

    class TTTObserver:
        def __init__(self, iig_obs_type, params):
            """Initializes an empty observation tensor."""
            if params:
                raise ValueError(f"Observation parameters not supported; passed {params}")
            shape = (1 + 2, rows, cols)  # (player, row, col), player = (opponent, current player, empty)
            self.tensor = np.zeros(np.prod(shape), np.float32)
            self.dict = { "observation": np.reshape(self.tensor, shape) }

        def one_hot(self, x):
            return np.identity(14)[x].flatten()

        def set_from(self, state, player):
            """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
            obs = self.dict["observation"]
            obs.fill(0)

            for row in range(rows):
                for col in range(cols):
                    cell_state = state.board[row, col]
                    cs = cell_state if cell_state == EMPTY else int(cell_state == player)
                    obs[cs, row, col] = 1

        def string_from(self, state, player):
            """Observation of `state` from the PoV of `player`, as a string."""
            return str(state)

    pyspiel.register_game(_GAME_TYPE, _TTTGame)


if __name__ == "__main__":
    #     $ python -m games.tic_tac_toe._pyspiel
    register_pyspiel(3, 5, 3, "ttt5x5-3")
    game = pyspiel.load_game("ttt5x5-3")

    state = game.new_initial_state()
    while not state.is_terminal():
        print(state.legal_actions())
        action = np.random.choice(state.legal_actions())
        print(state)
        print(f"{action=}")
        state.apply_action(action)
    print(state)
    print(state.returns())
