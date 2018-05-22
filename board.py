#!/usr/bin/python3

import numpy as np
from itertools import product


class Board:
    """ Tracks a single game state. """

    def __init__(self, side):
        """ Creates a board of side * side size. """
        assert (isinstance(side, int))
        self.side = side
        self.board = np.zeros((side, side), dtype=int)
        self.available = list(product(range(side), range(side)))

    def conv_one_hot(self):
        """ Returns the board state in one-hot encoding. The returned
        array has shape (1, 3, side, side) where each of the three channels
        represents either player or a free position.
        """
        b = self.board
        return np.expand_dims(np.stack((b == 0, b == 1, b == 2)), 0).astype('f4')

    def is_legal(self, x, y):
        """ Indicates if the given position is within the board. """
        return x >= 0 and x < self.side and y >= 0 and y < self.side

    def is_free(self, x, y):
        """ Indicates if the given position is unoccupied. """
        return self.board[x, y] == 0

    def place_move(self, x, y, player):
        """ Places a player move on the stated positon. Assert's that's OK. """
        assert (self.is_free(x, y))
        assert (player == 1 or player == 2)
        self.board[x, y] = player
        self.available.remove((x, y))

    def check_winner(self, required, x, y):
        """ Checks if move at (x, y) is part of a winning combination
        of required length. If so returns the player with that combination.
        If it is not, None is returned.
        """
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
