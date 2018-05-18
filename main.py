#!/usr/bin/python3

import argparse
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from board import Board

NET_PLAYER = 1
OTHER_PLAYER = 2

class Net(nn.Module):
    def __init__(self, board_side, hidden_layer=512):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(board_side ** 2 * 3, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, board_side ** 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, 0)

def train(board_side, win_row, game_count, model, device, optimizer):
    model.train()

    move_counts = []
    winners = []
    t0 = time()
    for game_ind in range(game_count):
        board = Board(board_side)
        move_outputs = []

        net_plays_next = bool(game_ind % 2)
        for move_ind in range(board_side ** 2):
            if net_plays_next:
                x = torch.from_numpy(board.flat_one_hot()).to(device)
                output = model(x)

                # Random selection of a legal move based on probs.
                mask = (board.board.flatten() == 0).astype('f4')
                probs = (output.detach().cpu().numpy() * mask)
                probs /= probs.sum()
                move = np.random.choice(probs.size, p=probs)

                board.place_move(move // board_side,
                                 move % board_side,
                                 NET_PLAYER)
                move_outputs.append(output[move])
            else:
                move = board.random_move()
                board.place_move(move[0], move[1], OTHER_PLAYER)

            net_plays_next = not net_plays_next
            winner = board.check_winner(win_row)
            if winner:
                move_counts.append(board.move_count())
                winners.append(winner)
                break
            if move_ind == board_side ** 2 - 1:
                winners.append(0)

        # Calculate losses and apply gradient
        net_won = winners[-1] == NET_PLAYER
        model.zero_grad()
        for move_output in move_outputs:
            if not net_won:
                move_output = 1 - move_output
            loss = -move_output.log()
            loss.backward()

        optimizer.step()

    print("Played", game_count, "games in ", time() - t0, " secs, with average",
          np.mean(move_counts), "moves")
    print("Net won ", (np.array(winners) == NET_PLAYER).mean())


def parse_args():
    parser = argparse.ArgumentParser(description='TicTacToe policy gradients')
    parser.add_argument('--board-side', type=int, default=3,
                        help='number of tiles on one side of the board')
    parser.add_argument('--win-row', type=int, default=3,
                        help='number of same-in-a-row for win')
    parser.add_argument('--games-per-update', type=int, default=64, metavar='N',
                        help='number of games to play per model update')
    parser.add_argument('--updates', type=int, default=100, metavar='N',
                        help='the number of net updates to perform')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    return parser.parse_args()

def main():
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net(args.board_side).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for _ in range(args.updates):
        train(args.board_side, args.win_row, args.games_per_update, model, device, optimizer)


if __name__ == '__main__':
    main()
