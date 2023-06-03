import numpy as np
from typing import Iterable

EMPTY = 2
PLAYERS_STR = ['X', 'O', ' ']


class TTT:
    def __init__(self, rows: int, cols: int, to_connect: int) -> None:
        self._next_player = 0
        self._scores = [0, 0]
        self._moves_played = 0
        self._rows = rows
        self._cols = cols
        self.board = np.full((rows, cols), EMPTY)
        self.to_connect = to_connect

    def is_full(self) -> bool:
        return self._moves_played >= self._rows * self._cols

    def to_str(self) -> str:
        lines = []
        lines.append("┏━━━" + "━━━".join("┳" * (self._cols - 1)) + "━━━┓")
        for i, row in enumerate(self.board):
            line = "┃"
            for e in row:
                line += " " + PLAYERS_STR[e] + " ┃"
            lines.append(line)
            if i != self._rows - 1:
                lines.append("┣━━━" + "━━━".join("╋" * (self._cols - 1)) + "━━━┫")
        lines.append("┗━━━" + "━━━".join("┻" * (self._cols - 1)) + "━━━┛")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.to_str()

    def __str__(self) -> str:
        return f"""{self.to_str()}\nScores {PLAYERS_STR[:2]}: {self._scores}"""

    def returns(self) -> tuple[int, int]:
        return self._scores

    def _check_line(self, x, y, dx, dy):
        mark = self.board[y, x]
        assert mark != EMPTY
        _found = 1  # current position
        for m in [-1, 1]:
            _x = x
            _y = y
            while True:
                _x += m * dx
                _y += m * dy
                if not (0 <= _x < self._cols) or not (0 <= _y < self._rows):
                    break
                if self.board[_y, _x] == mark:
                    _found += 1
                else:
                    break
        return _found >= self.to_connect

    def apply_action(self, action: tuple[int, int]) -> None:
        """
        Play the action for the current player.

        Arguments
        =========
            action: position to put the symbol on, (row, col)
        """
        assert not self.is_full()
        assert self.board[action] == EMPTY

        y, x = action
        mark = self._next_player
        self._moves_played += 1
        self.board[y, x] = mark

        # check all directions for a winning condition
        for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            if self._check_line(x, y, dx, dy):
                self._scores[self._next_player] += 1

        self._next_player = 1 - self._next_player

    def legal_actions(self) -> Iterable[tuple[int, int]]:
        """
        Return list of legal actions for current player.

        Returns
        =======
            list of (row, col) pairs
        """
        return [
            (r, c) for r in range(self._rows)
            for c in range(self._cols)
            if self.board[r, c] == EMPTY
        ]


if __name__ == "__main__":
    game = TTT(2, 2, 1)
    while not game.is_full():
        player = PLAYERS_STR[game._next_player]
        legal_actions = set(game.legal_actions())
        print(game)
        print(f"{legal_actions=}")
        row, col = eval(input(f"Enter next move (row, col) for {player}: "))
        game.apply_action((row, col))
    print(game)
    print(f"{game.returns()=}")
