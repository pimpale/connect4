# Connect4
## Purpose of this Tutorial
The prupose of this tutorial is to teach you how to create a maximally simple game-playing RL agent (for connect4 in this case).
I've found that other tutorials for RL often either abstract away the environment or have poor readability.

## Organization
In [Stage1](#Stage1) we will create a simple connect4 environment, a random agent, and a player agent, that lets you play against the random agent.

In [Stage2](#Stage2) we will create a minimax agent.

In [Stage3](#Stage3) we will define the kinds of features the connect4 agent sees, and create a simple actor and critic.

In [Stage4](#Stage4) we will calculate advantages using GAE (Generalized Advantage Estimation), define the training loop, and train the agent using PPO (Proximal Policy Optimization).

## Stage1

### Connect4 Environment
There's only one code block for you to complete. It's in the `step` function of the `Env` class. The `Env` class is the connect4 environment. Read through the rest of the code, as it's fairly straightforward.

In order to implement checking for the win condition efficiently, it is reccomended to use the scipy `convolve2d` function. You can read about it [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.convolve2d.html).

### Random Agent
The code is provided as an example.

### Human Agent
The code is provided as an example.

Once you're done, you should be able to play the game in the main notebook.ipynb.

## Stage2

### Minmax Agent
There's only one code block for you to complete. It's in the `minimax` function, used in the `MinmaxAgent` class. The `MinmaxAgent` class is the minimax agent. Read through the rest of the code, as it's fairly straightforward.


## Stage3

In stage 3, we create a MVP of the RL connect4 agent.
This is the most challenging stage, as there are a lot of moving parts that we need to add.

Here's the list of the parts we'll add in this stage:
1. Actor Network
2. Critic Network
3. Computing Value
4. Computing Advantage
5. Define Policy Gradient Loss
6. Training Actor and Critic


Note: We're purposely not implementing PPO yet. That comes in stage 4. Additionally, we're leaving off some optimizations in favor of simplicity.


### 1. Actor Definition
Go into `network.py` and fill out the missing sections in the `Actor` class.

### 2. Critic Definition
Go into `network.py` and fill out the missing sections in the `Critic` class.

### 3. Computing Value
Go into `network.py` and fill out the rest of the `compute_value` function.

### 4. Computing Advantage
Go into `network.py` and fill out the rest of the `compute_advantage` function.

### 5. Defining Policy Gradient Loss
Go into `network.py` and fill out the rest of the `compute_policy_gradient_loss` function.

### 6. Training Actor and Critic
Go into `network.py` and fill out the `train_policygradient` function.