# Connect4
## Purpose of this Tutorial
The prupose of this tutorial is to teach you how to create a maximally simple game-playing RL agent (for connect4 in this case).
I've found that other tutorials for RL often either abstract away the environment or have poor readability.

## Organization
In [Stage1](#Stage1) we will create a simple connect4 environment, a random agent, and a player agent, that lets you play against the random agent.

In [Stage1](#Stage2) we will create a minimax agent.

In [Stage3](#Stage3) we will define the kinds of features the connect4 agent sees, and create a simple actor and critic.

In [Stage4](#Stage4) we will calculate advantages using GAE (Generalized Advantage Estimation), define the training loop, and train the agent using PPO (Proximal Policy Optimization).

## Stage1

### Connect4 Environment