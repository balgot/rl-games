import time
import pygame
from ._game import EMPTY, PLAYERS_STR, TTT
from ._pyspiel import register_pyspiel
import pyspiel


CELL_SIZE = 100
FONT_SIZE = 50
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
SYMBOLS = PLAYERS_STR
COLORS = [RED, BLUE, WHITE]


class _InteractivePlay:
    def __init__(self, game_name: str, player1 = None, player2 = None, delay = 0):
        game = pyspiel.load_game(game_name)
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

                    # Check if the click was within the grid
                    if row < self.rows and col < self.cols:
                        if self.state.board[row, col] == EMPTY:
                            action = self.state.pos2action((row, col))
                            return action


def play_pygame(game: TTT, player1=None, player2=None, delay=0):
    igame = _InteractivePlay(game, player1, player2, delay)
    while not igame.is_terminal():
        igame.next_move()
    print(igame.state)
    print(igame.state.returns())


if __name__ == "__main__":
    register_pyspiel(4, 4, 3, "ttt")
    play_pygame("ttt")
