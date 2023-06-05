from ._game import EMPTY, PLAYER1, PLAYER1_HEAD, PLAYER2, PLAYER2_HEAD, FRUIT
import pygame
import time
import pyspiel
from ._pyspiel import register_pyspiel


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
    def __init__(self, game_name: str, player1, player2) -> None:
        game = pyspiel.load_game(game_name)
        self.state = game.new_initial_state()

        self.HEIGHT = self.state._game.height * CELL_SIZE
        self.WIDTH = self.state._game.width * CELL_SIZE

        pygame.init()
        self.game = game
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        self._players = [player1, player2]

    def is_terminal(self):
        return self.state.is_terminal()

    def render(self):
        self.screen.fill((255, 255, 255))

        for y in range(self.state._game.height):
            for x in range(self.state._game.width):
                pygame.draw.rect(
                    self.screen,
                    COLORS[self.state._game.board[y, x]],
                    (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    width=0
                )

        pygame.display.update()

    def next_move(self) -> None:
        self.render()
        human_actions = self._human_player()
        for ac, p in zip(human_actions, self._players):
            if p is None:
                action = ac
            else:
                action = p(self.state)
            print(["UP", "LEFT", "DOWN", "RIGHT"][action])
            self.state.apply_action(action)
        self.render()

    def _human_player(self):
        actions = [None, None]

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
                    actions[player] = action
        return actions


def play_pygame(game: str, player1=None, player2=None, delay=0.05):
    ip = _InteractiveSnakes(game, player1, player2)
    print("Initiated", ip)
    ip.render()
    time.sleep(delay)
    while not ip.is_terminal():
        print(ip.state._game)
        ip.next_move()
        time.sleep(delay)
    print(ip.state)
    print(ip.state.returns())


if __name__ == "__main__":
    import random
    register_pyspiel(5, 5, "my_snakes")
    RND = lambda state: random.choice(state.legal_actions())
    play_pygame("my_snakes", lambda _: 3, lambda _: 0, 1)
