import numpy as np
from collections import deque
import random

# directions
NO_DIR = (0, 0)
UP = (-1, 0)
LEFT = (0, -1)
DOWN = (1, 0)
RIGHT = (0, 1)

# game constants
NUM_PLAYERS = 2
ACTIONS = list(range(4))  # (up, left, down, right)
ACTION_TO_DIR = [UP, LEFT, DOWN, RIGHT]

# player encoding
EMPTY = 0
PLAYER1 = 1
PLAYER1_HEAD = 2
PLAYER2 = -1
PLAYER2_HEAD = -2
FRUIT = 3
PLAYERS = [PLAYER1, PLAYER2]

# helpers
def _is_empty(x):
    return x == EMPTY

def _is_head(x):
    return abs(x) == 2

# visual
SYMBOLS = {
    PLAYER1: "◼",
    PLAYER2: "◻",
    PLAYER1_HEAD: {
        NO_DIR: "◆",
        UP: "▲",
        RIGHT: "▶",
        LEFT: "◀",
        DOWN: "▼"
    },
    PLAYER2_HEAD: {
        NO_DIR: "◇",
        UP: "△",
        RIGHT: "▷",
        LEFT: "◁",
        DOWN: "▽"
    },
}


class Snakes:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        TOP_LEFT = (0, 0)
        BOTTOM_RIGHT = (height-1, width-1)

        self.board = np.full((height, width), EMPTY)
        self.fruit = None
        self.velocities = { p: NO_DIR for p in PLAYERS }
        self.alive = { p: True for p in PLAYERS }
        self.snakes = { p: deque([pos]) for p, pos in zip(PLAYERS, [TOP_LEFT, BOTTOM_RIGHT]) }

        # set up board
        for p, mark in zip([PLAYER1, PLAYER2], [PLAYER1_HEAD, PLAYER2_HEAD]):
            snake = self.snakes[p]
            assert len(snake) == 1
            self.board[snake[0]] = mark
        self._spawn_fruit()

    def __str__(self):
        return (
            f"{self.__class__.__name__}({self.height, self.width})\n"
            f"{self.to_str()}\n"
            f"{self.snakes}"
        )

    def to_str(self):
        lines = []
        lines.append("┏" + "━" * self.width + "┓")

        for row in self.board:
            line = "┃"
            for e in row:
                if _is_empty(e):
                    line += " "
                elif _is_head(e):
                    line += SYMBOLS[e][self.velocities[e // 2]]
                elif e == FRUIT:
                    line += "F"
                else:
                    line += SYMBOLS[e]
            line += "┃"
            lines.append(line)
        lines.append("┗" + "━" * self.width + "┛")

        return "\n".join(lines)

    def make_move(self, player: int, action: int):
        assert player in [PLAYER1, PLAYER2]
        assert action in ACTIONS
        self.velocities[player] = ACTION_TO_DIR[action]

    def _spawn_fruit(self):
        positions = [
            (r, c) for r in range(self.height)
                   for c in range(self.width)
                   if _is_empty(self.board[r, c])
        ]

        if not positions:
            self.fruit = None
            return

        self.fruit = random.choice(positions)
        self.board[self.fruit] = FRUIT

    def _is_collision(self, y, x, ignore=None):
        if ignore is None:
            ignore = {}

        return (
            y < 0 or x < 0
            or y >= self.height or x >= self.width
            or self.board[y, x] not in ignore
        )

    def _move_player(self, player, vel, snake: deque) -> bool:
        """Return true if crashes into the wall."""
        # if not moving, dont do anything - cant die
        if vel == (0, 0):
            return False

        dy, dx = vel
        y, x = snake[0]
        ny, nx = dy + y, dx + x
        _head = player * 2
        _body = player

        # if eating food, not much change
        if self.fruit == (ny, nx):
            self.board[y, x] = _body
            self.board[ny, nx] = _head
            snake.appendleft((ny, nx))
            self.fruit = None
            return False

        # otherwise pop tail and check collision
        snake.appendleft((ny, nx))
        self.board[snake[1]] = _body
        self.board[snake[-1]] = EMPTY
        snake.pop()

        if self._is_collision(ny, nx, ignore={EMPTY, FRUIT}):
            # print(f"[INFO] Collision: {(ny, nx)=}\t{snake=}")
            self.alive[player] = False
            return True

        # no collision - update board
        self.board[snake[0]] = _head
        return False

    def is_game_over(self):
        return not all(self.alive.values())

    def winner(self):
        if not self.is_game_over():
            return None
        return PLAYER1 if self.alive[PLAYER1] else PLAYER2 if self.alive[PLAYER2] else None

    def step(self):
        assert not self.is_game_over()

        # move players
        for player in PLAYERS:
            self._move_player(player, self.velocities[player], self.snakes[player])

        # check for collision between snakes
        for player, opp in zip(PLAYERS, PLAYERS[::-1]):
            snake = self.snakes[player]
            if snake[0] in self.snakes[opp]:
                self.alive[player] = False

        if not self.is_game_over() and self.fruit is None:
            self._spawn_fruit()


if __name__ == "__main__":
    game = Snakes(4, 4)
    print("One player moving game...")
    player = int(input("Enter player: "))
    assert player in (PLAYER1, PLAYER2)
    while not game.is_game_over():
        print(game)
        move = input("WASD: ").upper()
        move = "WASD".index(move)
        game.make_move(player, move)
        game.step()
    print(game)
    print(f"{game.winner()=}")
