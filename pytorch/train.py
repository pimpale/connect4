import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from torch import optim
import torch
import os
from scipy.special import softmax

import env
import network

BOARD_XSIZE=7
BOARD_YSIZE=6
DIMS=(BOARD_YSIZE,BOARD_XSIZE)

EPISODES_PER_AGENT = 50
TRAIN_EPOCHS = 500000
MODEL_SAVE_INTERVAL = 100
SUMMARY_STATS_INTERVAL = 10
RANDOM_SEED = 42

SUMMARY_DIR = './summary'
MODEL_DIR = './models'

# settings
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)


use_cuda = torch.cuda.is_available()
torch.manual_seed(RANDOM_SEED)

cuda = torch.device("cuda")
cpu = torch.device("cpu")

if use_cuda:
    device = cuda
else:
    device = cpu

# TODO: restore neural net parameters

actor = network.Actor(BOARD_XSIZE, BOARD_YSIZE).to(device)
critic = network.Critic(BOARD_XSIZE, BOARD_YSIZE).to(device)

actor_optimizer = optim.Adam(actor.parameters(), lr=network.ACTOR_LR)
critic_optimizer = optim.Adam(actor.parameters(), lr=network.CRITIC_LR)

# Get Writer
writer = SummaryWriter(log_dir=SUMMARY_DIR)

step=0


ACTOR_ID = np.int8(1)
OPPONENT_ID = np.int8(2)

summary_reward_buf:list[float] = []
for epoch in range(TRAIN_EPOCHS):
    s_batch:list[env.Observation] = []
    a_batch:list[env.Action] = []
    d_batch:list[env.Advantage] = []
    v_batch:list[env.Value] = []
    for _ in range(EPISODES_PER_AGENT):
        e = env.Env(DIMS)

        s_t:list[env.Observation] = []
        a_t:list[env.Action] = []
        r_t:list[env.Reward] = []
        
        actor_turn = True
        while not e.game_over():
            if actor_turn:
                obs = e.observe(ACTOR_ID)

                action_probs = actor.forward(network.obs_to_tensor(obs, device))[0].to(cpu).detach().numpy()
                if np.isnan(action_probs).any():
                    raise ValueError("NaN found!")

                action_logprobs = np.log(action_probs)

                # apply noise to probs
                noise = 0.1*np.random.gumbel(size=len(action_logprobs))
                adjusted_action_probs = softmax(action_logprobs + noise) 

                legal_mask = e.legal_mask() 

                chosen_action: env.Action = np.argmax(adjusted_action_probs*legal_mask)


                reward,_ = e.step(chosen_action, ACTOR_ID)

                s_t.append(obs)
                a_t.append(chosen_action)
                r_t.append(reward)
            else:
                legal_mask = e.legal_mask()
                action_prob = np.random.random(size=BOARD_XSIZE)
                chosen_action: env.Action = np.argmax(action_prob*legal_mask)
                e.step(chosen_action, OPPONENT_ID)
                # else:
                #     obs = e.observe(OPPONENT_ID)
                #     action_prob = actor.predict_batch([obs])[0]
                #     legal_mask = e.legal_mask() 
                #     chosen_action: env.Action = np.argmax(action_prob*legal_mask)
                #     e.step(chosen_action, OPPONENT_ID)

            # flip turn
            actor_turn = not actor_turn

        v_t = network.compute_value(r_t)
        d_t = network.compute_advantage(critic, s_t, r_t)


        # now update the minibatch
        s_batch += s_t
        a_batch += a_t
        d_batch += d_t
        v_batch += v_t
        summary_reward_buf.append(float(v_t[-1]))
        
    actor_loss, critic_loss = network.train(actor, critic, actor_optimizer, critic_optimizer, s_batch, a_batch, d_batch, v_batch)
    writer.add_scalar('actor_loss', actor_loss, step)
    writer.add_scalar('critic_loss', critic_loss, step)

    if epoch % SUMMARY_STATS_INTERVAL == 0:
        avg_reward = sum(summary_reward_buf)/len(summary_reward_buf)
        writer.add_scalar('avg_reward', avg_reward, step)
        # clear
        summary_reward_buf = []
    
    if epoch % MODEL_SAVE_INTERVAL == 0:
        # Save the neural net parameters to disk.
        actor_path = f"{SUMMARY_DIR}/nn_model_ep_{epoch}_actor.ckpt"
        critic_path = f"{SUMMARY_DIR}/nn_model_ep_{epoch}_critic.ckpt"
        torch.save(actor.state_dict(), actor_path)
        torch.save(critic.state_dict(), critic_path)
    
    step += 1
