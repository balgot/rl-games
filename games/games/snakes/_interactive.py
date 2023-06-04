from ._game import Snakes, EMPTY, PLAYER1, PLAYER1_HEAD, PLAYER2, PLAYER2_HEAD, FRUIT, _PLAYERS
import pygame
import time

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
GREEN_DARK = (100, 255, 100)
RED = (255, 0, 0)
RED_DARK = (255, 100, 100)
PURPLE = (255, 0, 255)


CELL_SIZE = 50
COLORS = {
    EMPTY: BLACK,
    FRUIT: PURPLE,
    PLAYER1: GREEN,
    PLAYER1_HEAD: GREEN_DARK,
    PLAYER2: RED,
    PLAYER2_HEAD: RED_DARK
}


class _InteractiveSnakes:
    def __init__(self, game: Snakes, player1, player2) -> None:
        self._players = [player1, player2]

        pygame.init()
        self.game = game
        self.HEIGHT = game.height * CELL_SIZE
        self.WIDTH = game.width * CELL_SIZE
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))

    def is_terminal(self):
        return self.game.is_game_over()

    def render(self):
        self.screen.fill((255, 255, 255))

        for y in range(self.game.height):
            for x in range(self.game.width):
                pygame.draw.rect(
                    self.screen,
                    COLORS[self.game.board[y, x]],
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    width=0
                )

        pygame.display.update()

    def next_move(self) -> None:
        self.game.step()
        self.render()
        self._human_player()
        for idx, p in zip(_PLAYERS, self._players):
            if p is not None:
                action = p(self.game) # TODO: use pyspiel.state
                self.game.make_move(idx, action)

    def _human_player(self):
        MAPPING = {
            pygame.K_w: ("W", 0),
            pygame.K_a: ("A", 0),
            pygame.K_s: ("S", 0),
            pygame.K_d: ("D", 0),

            pygame.K_UP: ("W", 1),
            pygame.K_LEFT: ("A", 1),
            pygame.K_DOWN: ("S", 1),
            pygame.K_RIGHT: ("D", 1),
        }

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key in MAPPING:
                action, player = MAPPING[event.key]
                if self._players[player] is None:
                    self.game.make_move(_PLAYERS[player], action)


def play_pygame(width: int, heaight: int, player1=None, player2=None, delay=0.05):
    game = Snakes(width, heaight)
    ip = _InteractiveSnakes(game, player1, player2)
    print("Initiated", ip)
    while not ip.is_terminal():
        ip.next_move()
        time.sleep(delay)
    print(game)
    print(game.winner())


if __name__ == "__main__":
    play_pygame(5, 5)
