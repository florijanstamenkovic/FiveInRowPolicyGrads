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
        assert (isinstance(size, int))
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.available = list(product(range(size), range(size)))

    def conv_one_hot(self):
        ''' Returns a numpy array of shape (1, 3, H, W). '''
        b = self.board
        return np.expand_dims(np.stack((b == 0, b == 1, b == 2)), 0).astype('f4')

    def flat_one_hot(self):
        ''' Returns a numpy array of shape (H * W, ). '''
        b = self.board
        return np.concatenate((b == 0, b == 1, b == 2)).flatten().astype('f4')

    def print_board(self):
        for row in range(self.size):
            if row != 0:
                print("-+" * (self.size - 1) + "-")
            print("|".join(Board.player_sym(x) for x in self.board[row]))

    def is_legal(self, x, y):
        return x >= 0 and x < self.size and y >= 0 and y < self.size

    def is_free(self, x, y):
        return self.board[x, y] == 0

    def place_move(self, x, y, player):
        assert (self.is_free(x, y))
        assert (player == 1 or player == 2)
        self.board[x, y] = player
        self.available.remove((x, y))

    def check_winner(self, required, x, y):
        ''' Checks if move at (x, y) is part of a winning combination
        of required length '''
        player = self.board[x, y]
        if player == 0:
            return None
        for axis in np.array(((1, 0), (0, 1), (1, 1), (-1, 1))):
            found = 1
            for direction in [axis, -axis]:
                for pos in [
                        np.array([x, y]) + direction * d
                        for d in range(1, required)
                ]:
                    if self.is_legal(
                            pos[0],
                            pos[1]) and self.board[pos[0], pos[1]] == player:
                        found += 1
                    else:
                        break

            if found >= required:
                return player
        return None

    def move_count(self):
        return self.size**2 - len(self.available)
