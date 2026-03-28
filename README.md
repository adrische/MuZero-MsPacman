# Step-by-step Reimplementation Attempt of MuZero for Ms Pacman

> Work in progress. Feedback welcome.


## Contents

1. [Dyna-Q](#dyna-q-notebook)
2. [Deep-Q-Network (DQN)](#dqn-notebook)
3. [Monte-Carlo Tree Search (MCTS)](#mcts-notebook)
4. [MuZero](#muzero)
5. [References](#references)



## Dyna-Q [Notebook](notebooks/dyna-q.ipynb) 

Dyna-Q is a Q-learning algorithm with a planning component that iterates additional Q-learning steps based on previously encountered state-action-reward-next state transitions. We implement Tabular Dyna-Q from Chapter 8 of Sutton & Barto (Example 8.1) with a gridworld environment.


## DQN [Notebook](notebooks/DQN.ipynb)

DQN trains a network to predict Q-values based on previously seen transitions sampled from a replay buffer in a supervised fashion.

| Environment | Random policy | Trained policy | Training details |
| ----------- | ------------- | -------------- | ---------------- |
| Pong | ![](assets/DQN_ALEPong-v5_random.gif) | Random action during evaluation with 0% vs 5% probability. Scores: 5:21 vs 8:21<br>![](assets/DQN_ALEPong-v5_0epochs_Wed_Mar_11_07:30:02_2026_eps0_0.gif) ![](assets/DQN_ALEPong-v5_0epochs_Wed_Mar_11_07:30:02_2026_eps0_05.gif) | 10 mio frames | 


## MCTS [Notebook](notebooks/MCTS-reproduction-of-MCTX-visualization-demo.ipynb)

Monte-Carlo tree search is a seach algorithm that selects at each step the most promising action, in terms of how good actions are expected to be vs. how much uncertainty there is. New actions are initialized by accessing either the environment or a model thereof, while existing Q-value estimates are used within the tree. We implement a naive Python version of the MCTS algorithm used by MuZero, and compare its output with the faster JAX implementation released by DeepMind, [MCTX](https://github.com/google-deepmind/mctx).

![](assets/reproduction_mctx_visualization_demo.png)


## MuZero

Work in progress: As a first step, I'm implementing changes to the DQN code only based on the MuZero paper, with the goal of having a scaled-down version of MuZero that can demonstrate improvements to DQN on Ms Pacman. I won't be looking at the published pseudocode for now. Afterwards, I'll review my implementation against the pseudocode.



## References

### Main

[Reinforcement Learning: An Introduction (Sutton & Barto)](http://incompleteideas.net/book/the-book-2nd.html)

[Playing Atari with Deep Reinforcement Learning (DQN Arxiv 2013)](https://arxiv.org/abs/1312.5602)

[Human-level control through deep reinforcement learning (DQN Nature 2015)](https://www.nature.com/articles/nature14236)

[Mastering the game of Go with deep neural networks and tree search (AlphaGo)](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf)

[Mastering the Game of Go without Human Knowledge (AlphaGo Zero)](https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf)

[Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero)](https://arxiv.org/pdf/1712.01815)

[Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero)](https://arxiv.org/abs/1911.08265)

[MuZero Pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)

[Monte Carlo tree search in JAX (MCTX)](https://github.com/google-deepmind/mctx)


### Related

[Deep Reinforcement Learning and the Deadly Triad](https://arxiv.org/pdf/1812.02648) (function approximation, off-policy learning, and bootstrapping)



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
