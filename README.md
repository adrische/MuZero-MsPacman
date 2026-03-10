# Step-by-step Reimplementation Attempt of MuZero for Ms Pacman

> Work in progress. Feedback welcome.


## Contents

1. [Dyna-Q](#dyna-q)
2. [Deep-Q-Network (DQN)](#dqn)
3. [Monte-Carlo Tree Search (MCTS)](#mcts)
4. [MuZero](#muzero)


## Dyna-Q [Notebook](notebooks/dyna-q.ipynb) 

Dyna-Q is a Q-learning algorithm with a planning component that iterates additional Q-learning steps based on previously encountered state-action-reward-next state transitions. We implement Tabular Dyna-Q from Chapter 8 of Sutton & Barto (Example 8.1) with a gridworld environment.


## DQN [Notebook](notebooks/DQN.ipynb)

DQN trains a network to predict Q-values based on previously seen transitions sampled from a replay buffer in a supervised fashion.

| Environment | Random policy | Trained policy | Training details |
| ----------- | ------------- | -------------- | ---------------- |
| Pong | ![](assets/DQN_ALEPong-v5_random.gif) | Still running, maximum evaluation reward so far: 12 | 10 mio frames | 


## MCTS [Notebook](notebooks/MCTS-reproduction-of-MCTX-visualization-demo.ipynb)

Monte-Carlo tree search is a seach algorithm that selects at each step the most promising action, in terms of how good actions are expected to be vs. how much uncertainty there is. New actions are initialized by accessing either the environment or a model thereof, while existing Q-value estimates are used within the tree. We implement a naive Python version of the MCTS algorithm used by MuZero, and compare its output with the faster JAX implementation released by DeepMind, [MCTX](https://github.com/google-deepmind/mctx).

![](assets/reproduction_mctx_visualization_demo.png)


## MuZero

Work in progress








## TODO

- [x] Dyna-Q
- [x] DQN 
  - [x] Replay buffer
  - [x] Atari environment
  - [x] Neural network, stochastic gradient descent
  - [x] Training loop
  - [x] Signs of life :)
  - [ ] GPU
  - [x] Debug!
  - [ ] Remaining details from both DQN papers
  - [x] Run for the full number of frames
- [ ] MuZero
  - [x] Monte-Carlo tree search
    - [ ] Does it work with tensors / batches
  - [ ] Other changes to DQN
    - [ ] Different loss
    - [ ] TD-targets
    - [ ] Non-uniform sampling from replay buffer
    - [ ] ...
