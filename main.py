#!/usr/bin/python3

import argparse
import logging
import math
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
    @staticmethod
    def valid_size(in_size, field, stride):
        ''' Calculates conv otput for given input size, field size and stride. '''
        return math.ceil((in_size - field + 1) / stride)

    def __init__(self, board_side, conv1=(32, 2, 1), conv2=(64, 2, 2), fc1=256):
        super(ConvNet, self).__init__()
        self.pre_fc_side = board_side
        self.conv1 = nn.Conv2d(
            3, conv1[0], kernel_size=conv1[1], stride=conv1[2])
        self.pre_fc_side = ConvNet.valid_size(self.pre_fc_side, *conv1[1:])
        self.conv2 = nn.Conv2d(
            conv1[0], conv2[0], kernel_size=conv2[1], stride=conv2[2])
        self.pre_fc_side = ConvNet.valid_size(self.pre_fc_side, *conv2[1:])
        self.fc1 = nn.Linear(self.pre_fc_side ** 2 * conv2[0], fc1)
        self.fc2 = nn.Linear(fc1, board_side ** 2)
        print(self)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, 0)


def play_game(model, opponent, net_plays_next, args, device):
    """
    Plays a single game. One player is the given model,
    the other depends on the args.
    Returns (winner, model_move_outputs)
    """
    board = Board(args.board_side)
    move_outputs = []

    for move_ind in range(args.board_side ** 2):
        move = None
        if net_plays_next:
            x = torch.from_numpy(board.conv_one_hot()).to(device)
            output = model(x)

            # Random selection of a legal move based on probs.
            mask = (board.board.flatten() == 0).astype('f4')
            probs = (output.detach().cpu().numpy() * mask)
            probs /= probs.sum()
            move = np.random.choice(probs.size, p=probs)
            move_outputs.append(output[move])
            move = move // args.board_side, move % args.board_side
            board.place_move(move[0], move[1], NET_PLAYER)
        else:
            move = opponent(OTHER_PLAYER, board)
            board.place_move(move[0], move[1], OTHER_PLAYER)

        net_plays_next = not net_plays_next
        winner = board.check_winner(args.win_row, move[0], move[1])
        if winner is None and move_ind == args.board_side ** 2 - 1:
            winner = 0
        if winner is not None:
            return winner, move_outputs


def train(args, model, device, optimizer):
    model.train()

    winners = []
    t0 = time()
    move_counts = []
    opponent = player.for_name(args.opponent)
    for game_ind in range(args.episodes_per_update):
        winner, move_outputs = play_game(
            model, opponent, bool(game_ind % 2), args, device)
        winners.append(winner)
        move_counts.append(len(move_outputs) * 2)

        # Calculate losses and apply gradient
        model.zero_grad()
        for move_output in move_outputs:
            if not winner == NET_PLAYER:
                move_output = 1 - move_output
            loss = -move_output.log()
            loss.backward()

        optimizer.step()

    logging.info("Played %d games in %.2f secs, average %.2f moves",
                 args.episodes_per_update, time() - t0, np.mean(move_counts))
    winners = np.array(winners)
    logging.info("Net won %d (%.2f), tie: %d, lost %d",
                 (winners == NET_PLAYER).sum(),
                 (winners == NET_PLAYER).mean(),
                 (winners == 0).sum(),
                 (winners == OTHER_PLAYER).sum())


def evaluate(args, model, device):
    with torch.no_grad():
        for name, opponent in player.all_players().items():
            logging.info("Evaluating against '%s':", name)
            winners = []
            for game_ind in range(args.eval_episodes):
                winner, _ = play_game(model, opponent, bool(game_ind % 2), args, device)
                winners.append(winner)

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
    parser.add_argument('--opponent', default='random',
                        help='the opponent used during training')

    parser.add_argument('--episodes-per-update', type=int, default=128, metavar='N',
                        help='number of games to play per model update')
    parser.add_argument('--updates', type=int, default=100, metavar='N',
                        help='the number of net updates to perform')
    parser.add_argument('--eval-episodes', type=int, default=256,
                        help='how many episodes are played in evaluation')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    model = ConvNet(args.board_side).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

    for _ in range(args.updates):
        train(args, model, device, optimizer)
    evaluate(args, model, device)


if __name__ == '__main__':
    main()
