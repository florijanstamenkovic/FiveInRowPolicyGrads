#!/usr/bin/python3

import argparse
import logging
from itertools import count

import board
import player


def main():
    parser = argparse.ArgumentParser(description='TicTacToe policy gradients')
    parser.add_argument('--board-side', type=int, default=3,
                        help='number of tiles on one side of the board')
    parser.add_argument('--win-row', type=int, default=3,
                        help='number of same-in-a-row for win')
    parser.add_argument('--player', default='random',
                        help='the player used during testing')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    b = board.Board(args.board_side)
    winner = None
    for move_ind in count():
        current = 1 + move_ind % 2
        move = player.for_name(args.player)(current, b)
        b.place_move(*move, current)
        logging.info("Board after move %d", move_ind)
        b.print_board()
        winner = b.check_winner(args.win_row)
        if winner is None and len(b.available) == 0:
            winner = 0
        if winner is not None:
            print("Winner:", winner)
            break


if __name__ == '__main__':
    main()
