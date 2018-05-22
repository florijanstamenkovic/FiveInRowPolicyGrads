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

# Board values for the trained model and the opponent
# used for training.
MODEL = 1
OPPONENT = 2


def win_info_to_string(win_info):
    """ Formats a list of ints (game winners) into a summary. """
    win_info = np.array(win_info)
    return "model won %d/%d (%.2f), tie: %d, lost %d" % (
        (win_info == MODEL).sum(), win_info.size,
        (win_info == MODEL).mean(),
        (win_info == 0).sum(),
        (win_info == OPPONENT).sum())


class ConvNet(nn.Module):
    def __init__(self, board_side):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.convTrans = nn.ConvTranspose2d(64, 1, 4)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.convTrans(x).exp()
        return (x / x.sum()).view(-1)


def play_game(model, opponent, net_plays_next, args, device):
    """ Plays a single game. One player is the given model, the other
    is the given opponent. Returns (winner, chosen_move_probabilities).
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
            board.place_move(move[0], move[1], MODEL)
        else:
            move = opponent(OPPONENT, board)
            board.place_move(move[0], move[1], OPPONENT)

        net_plays_next = not net_plays_next
        winner = board.check_winner(args.win_row, move[0], move[1])
        if winner is None and move_ind == args.board_side ** 2 - 1:
            winner = 0
        if winner is not None:
            return winner, move_outputs


def update(args, model, device, optimizer):
    """ Performs a single update of the given model. """
    model.train()

    winners = []
    episodes = []
    opponent = player.for_name(args.opponent)
    for game_ind in range(args.episodes_per_update):
        winner, chosen_move_probs = play_game(
            model, opponent, bool(game_ind % 2), args, device)
        winners.append(winner)
        episodes.append(chosen_move_probs)

    # Calculate losses and apply gradient
    model.zero_grad()
    total_loss = 0
    for winner, chosen_move_probs in zip(winners, episodes):
        for chosen_move_prob in chosen_move_probs:
            prob = chosen_move_prob
            if winner != MODEL:
                prob = 1 - chosen_move_prob
            loss = -prob.log()
            loss.backward()
            total_loss += loss.detach().to(device)
    optimizer.step()

    return winners, total_loss


def evaluate(args, model, device):
    """ Evaluates the given model against all opponents. """
    with torch.no_grad():
        for name, opponent in player.all_players().items():
            logging.info("Evaluating against '%s':", name)
            winners = []
            for game_ind in range(args.eval_episodes):
                winner, _ = play_game(
                    model, opponent, bool(game_ind % 2), args, device)
                winners.append(winner)

            logging.info(win_info_to_string(winners))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Five-in-a-row policy gradients')
    parser.add_argument('--board-side', type=int, default=12,
                        help='number of tiles on one side of the board')
    parser.add_argument('--win-row', type=int, default=5,
                        help='number of same-in-a-row for win')
    parser.add_argument('--opponent', default='mixed',
                        choices=['random', 'greedy', 'mixed'],
                        help='the opponent used during training')

    parser.add_argument('--episodes-per-update', type=int, default=32, metavar='N',
                        help='number of games to play per model update')
    parser.add_argument('--updates', type=int, default=256, metavar='N',
                        help='the number of net updates to perform')
    parser.add_argument('--eval-episodes', type=int, default=128,
                        help='how many episodes are played in evaluation')

    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA training')

    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    model = ConvNet(args.board_side).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)

    logging.info("Eval : ---------------")
    evaluate(args, model, device)

    logging.info("Training : ---------------")
    t0 = time()
    for update_ind in range(args.updates):
        t1 = time()
        win_info, total_loss = update(args, model, device, optimizer)
        if update_ind % (args.updates // 20) != 0:
            continue
        logging.info("Update %d/%d: %d episodes in %.2f secs, %s, loss: %.2f",
                     update_ind, args.updates,
                     args.episodes_per_update, time() - t1,
                     win_info_to_string(win_info), total_loss)
    logging.info("Total training time: %.2fsec", time() - t0)

    logging.info("Eval : ---------------")
    evaluate(args, model, device)


if __name__ == '__main__':
    main()
