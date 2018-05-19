#!/usr/bin/python3

from random import randint
from itertools import product


def random(player, board):
    assert (board.available)
    return board.available[randint(0, len(board.available) - 1)]


def greedy(player, board):
    ''' Chooses a move adjacent to the existing move of the same player.
    If there are multiple such moves one is chosen at random. '''

    def adjacent_legal_free_tiles(x, y):
        for i, j in product([x - 1, x, x + 1], [y - 1, y, y + 1]):
            if board.is_legal(i, j) and board.is_free(i, j):
                yield (i, j)

    moves = []
    for x, y in product(range(board.size), repeat=2):
        if board.board[x, y] == player:
            moves.extend(adjacent_legal_free_tiles(x, y))

    if moves:
        return moves[randint(0, len(moves) - 1)]

    return random_move(player, board)

def for_name(name):
    if name == 'random':
        return random
    if name == 'greedy':
        return greedy
    raise Exception("Unknown opponent name: " + name)


