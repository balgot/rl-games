import numpy as np
from collections import deque
import random


EMPTY = 0

# players are 1, -1; heads are 2, -2
NUM_PLAYERS = 2
PLAYER1 = 1
PLAYER1_HEAD = 2
PLAYER2 = -1
PLAYER2_HEAD = -2
FRUIT = 3

_PLAYERS = [PLAYER1, PLAYER2]

def _is_empty(x):
    return x == EMPTY

def _is_head(x):
    return abs(x) == 2

# directions
UP = (-1, 0)
LEFT = (0, -1)
DOWN = (1, 0)
RIGHT = (0, 1)

ACTIONS = {
    "W": UP,
    "A": LEFT,
    "S": DOWN,
    "D": RIGHT
}

SYMBOLS = {
    PLAYER1: "◼",
    PLAYER2: "◻",
    PLAYER1_HEAD: {
        (0, 0): "◆",
        UP: "▲",
        RIGHT: "▶",
        LEFT: "◀",
        DOWN: "▼"
    },
    PLAYER2_HEAD: {
        (0, 0): "◇",
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

        self.board = np.full((height, width), EMPTY)
        self.fruit = None
        self.velocities = {
            PLAYER1: (0, 0),
            PLAYER2: (0, 0)
        }
        self.snakes = {
            PLAYER1: deque([(0, 0)]),
            PLAYER2: deque([(height-1, width-1)])
        }
        self.alive = {
            PLAYER1: True,
            PLAYER2: True
        }

        # set up board
        for p, mark in zip([PLAYER1, PLAYER2], [PLAYER1_HEAD, PLAYER2_HEAD]):
            snake = self.snakes[p]
            assert len(snake) == 1
            self.board[snake[0]] = mark
        self._spawn_fruit()

    def __str__(self):
        return f"{self.to_str()}\n{self.snakes}"

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

    def make_move(self, player: int, action: str):
        assert player in [PLAYER1, PLAYER2]
        assert action in ACTIONS
        self.velocities[player] = ACTIONS[action]

    def _spawn_fruit(self):
        positions = [(r, c) for r in range(self.height) for c in range(self.width) if self.board[r, c] == EMPTY]
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

    def _move_snake(self, player, vel, snake: deque):
        """Return True if we are fruit."""
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
            return True

        # otherwise pop tail and check collision
        # update head
        snake.appendleft((ny, nx))
        self.board[snake[1]] = _body
        self.board[snake[-1]] = EMPTY
        snake.pop()

        # print("Check collistion:", {EMPTY, FRUIT, _head}, (ny, nx))
        if self._is_collision(ny, nx, ignore={EMPTY, FRUIT, _head}):
            # print(f"[INFO] Collision: {(ny, nx)=}\t{snake=}")
            self.alive[player] = False
        else:
            self.board[snake[0]] = _head
        return False

    def is_game_over(self):
        return not all(self.alive.values())

    def winner(self):
        assert self.is_game_over()
        return PLAYER1 if self.alive[PLAYER1] else PLAYER2 if self.alive[PLAYER2] else None

    def step(self):
        # move the players
        fruit_was_eaten = False
        for player in _PLAYERS:
            fruit_was_eaten |= self._move_snake(player, self.velocities[player], self.snakes[player])

        if not self.is_game_over() and fruit_was_eaten:
            self._spawn_fruit()


if __name__ == "__main__":
    game = Snakes(4, 4)
    player = int(input("Enter player: "))
    assert player in (PLAYER1, PLAYER2)
    while not game.is_game_over():
        print(game)
        move = input("WASD: ").upper()
        game.make_move(player, move)
        game.step()
    print(game)
    print(game.winner())
