# Five-in-a-row policy gradient reinforcement learning

Make a neural net learn how to play five-in-a-row (also known as "Gomoku"). Use
policy gradient based reinforcement learning against two simple conventional
algorithm opponents.

## Requirements
* python3
* numpy
* torch

## Usage
```
# list arguments
python3 main.py --help

# start training with default params
# (10x10 board, 5-in-a-row, 256 updates, no CUDA)
python3 main.py
```

Board size, the row size required to win, training opponent type, training
duration and some hyperparams can be tweaked.

## Opponents
* random - places a move on a random available position
* greedy - places a move next to it's own piece, randomly

## Model
The neural net is defined in `main.py:ConvNet`. It's a convolutional net that
does two convolutions and then a transposed convolution to get a single channel
output with a board-sized grid. This is then turned into probabilities in a 2D
softmax way.

Illegal moves are not dealt with by the model, their probabilities are zeroed
after model output. It's reasonable to assume that after training the net should
not give them much probability. 

## CUDA
In the current implementation training with CUDA is actually slower. I've had
versions in which it was substantially faster (without episode accumulation into
a single update). I think it's to do with backprop accumulation, need to look
into it. To enable CUDA use `--cuda`.
