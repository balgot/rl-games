# Comparison of RL Algorithms on Simple Games

This repository contains a code and comparison of two RL algorithms:
* *AlphaZero*
* *Deep Q Networks*

on these two-player games:
* *Nim* (from `open_spiel` library)
* *2 player snakes* (see `games/games/snakes`)
* *generalised tic tac toe* (see `games/games/ttt`)


## Results

The results of the comparison are present in these notebooks:
* [AlphaZero](https://colab.research.google.com/drive/1l9sGcW466SBNRLsl0KvqVVt4NDJhBShi?usp=sharing)
* [DQN](https://colab.research.google.com/drive/1E-h5Xd3zd4BHzPwxNXf8lGGncJ0dfA29)
* [Comparison of AlphaZero and DQN](https://colab.research.google.com/drive/1YZE148MIXglvRXeyBv69KxAUfssOgZ9-?usp=sharing)


## How to run

1. make sure you have Python <= 3.11 and >= 3.9 (we tested on 3.10)
2. install requirements - either from `requirements.txt`, or:
    * numpy
    * open_spiel
    * pandas
    * [azero](https://github.com/balgot/alpha_zero_os)
    * pygame
    * wandb
3. install games package:
    `pip install games/`


## What is where

* `alpha_zero/` - contains training and evaluation scripts for *AlphaZero*
algorithm, and the evaluation results against random and MCTS agents;

* `dqn/` - contains training and evalution scripts for *DQNs*, also with
the evaluation results;

* `games/` - contains implementations of the studied games

