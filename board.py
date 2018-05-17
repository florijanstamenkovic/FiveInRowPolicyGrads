#!/usr/bin/python3

import numpy as np
from itertools import product

class Board:

    @staticmethod
    def player_sym(ind):
        if ind == 0:
            return " "
        if ind == 1:
            return "X"
        if ind == 2:
            return "O"

    def __init__(self, size):
        assert(isinstance(size, int))
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.available = list(product(range(size), range(size)))

    def print_board(self):
        for row in range(self.size):
            if row != 0:
                print("-+" * (self.size - 1) + "-")
            print ("|".join(Board.player_sym(x) for x in self.board[row]))

    def is_free(self, x, y):
        return self.board[x, y] == 0

    def place_move(self, x, y, player):
        assert(self.is_free(x, y))
        assert(player == 1 or player == 2)
        self.board[x, y] = player
        self.available.remove((x, y))

    def random_move(self):
        assert(self.available)
        return self.available[np.random.randint(len(self.available))]

    def check_winner(self, required):
        for i, j in product(range(self.size), range(self.size)):
            player = self.board[i, j]
            if player == 0:
                continue
            for direction in np.array(((1, 0), (0, 1), (1, 1), (-1, 1))):
                win = True
                for pos in [np.array([i, j]) + direction * d for d in range(required)]:
                    invalid = (pos < 0).sum() or (pos >= self.size).sum()
                    if invalid:
                        win = False
                        break
                    if self.board[pos[0], pos[1]] != player:
                        win = False
                if win:
                    return player
        return None


# Simple demo/test of Board class functionalities.
def main():
    size =5
    b = Board(size)
    for i in range(size ** 2):
        move = b.random_move()
        print("Board after move: ", move)
        b.place_move(*move, i % 2 + 1)
        b.print_board()
        winner = b.check_winner(3)
        if (winner):
            print("Winner:", winner)
            break;


if __name__ == '__main__':
    main();
