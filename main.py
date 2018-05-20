#!/usr/bin/python3

import argparse
import logging
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from board import Board
import player

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


class ConvNet(nn.Module):
    def __init__(self, board_side, conv1=64, fc1=256):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, conv1, kernel_size=2)
        self.fc1 = nn.Linear((board_side - 1) ** 2 * conv1, fc1)
        self.fc2 = nn.Linear(fc1, board_side ** 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, 0)


def train(args, model, device, optimizer):
    model.train()

    move_counts = []
    winners = []
    t0 = time()
    for game_ind in range(args.games_per_update):
        board = Board(args.board_side)
        move_outputs = []

        net_plays_next = bool(game_ind % 2)
        for move_ind in range(args.board_side ** 2):
            if net_plays_next:
                x = torch.from_numpy(board.conv_one_hot()).to(device)
                output = model(x)

                # Random selection of a legal move based on probs.
                mask = (board.board.flatten() == 0).astype('f4')
                probs = (output.detach().cpu().numpy() * mask)
                probs /= probs.sum()
                move = np.random.choice(probs.size, p=probs)

                board.place_move(move // args.board_side,
                                 move % args.board_side,
                                 NET_PLAYER)
                move_outputs.append(output[move])
            else:
                move = player.for_name(args.opponent)(OTHER_PLAYER, board)
                board.place_move(move[0], move[1], OTHER_PLAYER)

            net_plays_next = not net_plays_next
            winner = board.check_winner(args.win_row)
            if winner is None and move_ind == args.board_side ** 2 - 1:
                winner = 0
            if winner is not None:
                move_counts.append(board.move_count())
                winners.append(winner)
                break

        # Calculate losses and apply gradient
        net_won = winners[-1] == NET_PLAYER
        model.zero_grad()
        for move_output in move_outputs:
            if not net_won:
                move_output = 1 - move_output
            loss = -move_output.log()
            loss.backward()

        optimizer.step()

    logging.info("Played %d games in %.2f secs, average %.2f moves",
                 args.games_per_update, time() - t0, np.mean(move_counts))
    winners = np.array(winners)
    logging.info("Net won %d (%.2f), tie: %d, lost %d",
                 (winners == NET_PLAYER).sum(),
                 (winners == NET_PLAYER).mean(),
                 (winners == 0).sum(),
                 (winners == OTHER_PLAYER).sum())


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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--opponent', default='random',
                        help='the opponent used during training')
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    model = ConvNet(args.board_side).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    for _ in range(args.updates):
        train(args, model, device, optimizer)


if __name__ == '__main__':
    main()
