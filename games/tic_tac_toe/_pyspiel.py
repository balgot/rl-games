import sys
import time

import numpy as np
import pygame
import pyspiel
from ._game import _EMPTY, _PLAYERS_STR, TTT


CELL_SIZE = 100
FONT_SIZE = 50
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
SYMBOLS = _PLAYERS_STR
COLORS = [RED, BLUE, WHITE]


class _InteractivePlay:
    def __init__(self, game: TTT, player1 = None, player2 = None, delay = 0):
        self.state = game.new_initial_state()
        self.rows = self.state._game._rows
        self.cols = self.state._game._cols

        self.HEIGHT = self.rows * CELL_SIZE
        self.WIDTH = self.cols * CELL_SIZE + 2 * CELL_SIZE

        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self.font = pygame.font.Font(None, FONT_SIZE)
        self._delay = delay
        self._players = [player1, player2]
        self.game = game

    def is_terminal(self):
        return self.state.is_terminal()

    def _render_text(self, text, cx, cy, color):
        number_surface = self.font.render(str(text), True, color)
        number_rect = number_surface.get_rect(center=(cx, cy))
        self.screen.blit(number_surface, number_rect)

    def _update_display(self):
        self.screen.fill(WHITE)
        # draw the board
        for row in range(self.rows):
            for col in range(self.cols):
                pygame.draw.rect(self.screen, BLACK, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE), width=1)
                cx = col * CELL_SIZE + CELL_SIZE // 2
                cy = row * CELL_SIZE + CELL_SIZE // 2
                _occupancy = self.state.board[row, col]
                self._render_text(SYMBOLS[_occupancy], cx, cy, COLORS[_occupancy])

        # draw the score
        cx = self.WIDTH - CELL_SIZE // 2
        cy = self.HEIGHT // 2
        pygame.draw.rect(self.screen, BLACK, (self.WIDTH - CELL_SIZE, 0, CELL_SIZE, self.HEIGHT))
        _score = f"{self.state._game._scores[0]}-{self.state._game._scores[1]}"
        self._render_text(_score, cx, cy, WHITE)

        # Update the display
        pygame.display.update()

    def next_move(self) -> None:
        self._update_display()
        player_to_play = self.state._game._next_player
        assert player_to_play in (0, 1)
        play_fn = self._players[player_to_play]

        if play_fn is not None:
            action = play_fn(self.state)
        else:
            action = self._get_human_action()

        self.state.apply_action(action)
        self._update_display()
        if self._delay > 0:
            time.sleep(self._delay)

    def _get_human_action(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise RuntimeError("Closing the window during the game")
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    x, y = event.pos
                    row = y // CELL_SIZE
                    col = x // CELL_SIZE

                    print(row, self.rows, col, self.cols)

                    # Check if the click was within the grid
                    if row < self.rows and col < self.cols:
                        if self.state.board[row, col] == _EMPTY:
                            action = self.state.pos2action((row, col))
                            return action


def register_pyspiel(rows: int, cols: int, to_connect: int, name: str):
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

        def interactive_play(self, player1=None, player2=None):
            ip = _InteractivePlay(self, player1, player2)
            while not ip.is_terminal():
                ip.next_move()
            print(ip.state)
            print(ip.state.returns())

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
            return f"{_PLAYERS_STR[player]}{self.action2pos(action)}"

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
            # The observation should contain a 1-D tensor in `self.tensor` and a
            # dictionary of views onto the tensor, which may be of any shape.
            # Here the observation is indexed `(cell state, row, column)`.
            shape = (1 + 2, rows, cols)  # (player, row, col)
            self.tensor = np.zeros(np.prod(shape), np.float32)
            self.dict = { "observation": np.reshape(self.tensor, shape) }

        def one_hot(self, x):
            return np.identity(14)[x].flatten()

        def set_from(self, state, player):
            """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
            del player # unused
            # We update the observation via the shaped tensor since indexing is more
            # convenient than with the 1-D tensor. Both are views onto the same memory.
            obs = self.dict["observation"]
            obs.fill(0)
            for row in range(rows):
                for col in range(cols):
                    cell_state = state.board[row, col]
                    obs[cell_state, row, col] = 1

        def string_from(self, state, player):
            """Observation of `state` from the PoV of `player`, as a string."""
            return str(state)

    pyspiel.register_game(_GAME_TYPE, _TTTGame)


if __name__ == "__main__":
    register_pyspiel(3, 5, 3, "ttt5x5-3")
    game = pyspiel.load_game("ttt5x5-3")

    if "--interactive" in sys.argv:
        game.interactive_play()
    else:
        state = game.new_initial_state()
        while not state.is_terminal():
            print(state.legal_actions())
            action = np.random.choice(state.legal_actions())
            print(state)
            print(f"{action=}")
            state.apply_action(action)
        print(state)
        print(state.returns())
