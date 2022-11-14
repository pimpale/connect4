from io import TextIOWrapper
import multiprocessing as mp
import numpy as np
import numpy.typing as npt
import logging
import tensorflow as tf
import os
import sys
import network
import env

BOARD_XSIZE=7
BOARD_YSIZE=6
DIMS=(BOARD_YSIZE,BOARD_XSIZE)

BATCH_SIZE = 10
NUM_AGENTS = 2
TRAIN_EPOCHS = 500000
MODEL_SAVE_INTERVAL = 10
RANDOM_SEED = 42

SUMMARY_DIR = './summary'
MODEL_DIR = './models'
TRAIN_TRACES = './train/'
TEST_LOG_FOLDER = './test_results/'
LOG_FILE = SUMMARY_DIR + '/log'

# settings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# create result directory
if not os.path.exists(SUMMARY_DIR):
    os.makedirs(SUMMARY_DIR)

def testing(epoch:int, nn_actor_model_path:str, nn_critic_model_path:str, log_file:TextIOWrapper):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    #os.system('mkdir ' + TEST_LOG_FOLDER)
    if not os.path.exists(TEST_LOG_FOLDER):
        os.makedirs(TEST_LOG_FOLDER)
    # run test script
    os.system(f'python test.py {nn_actor_model_path} {nn_critic_model_path}')
    # append test performance to the log
    rewards_list:list[float] = []
    entropies_list:list[float] = []
    test_log_files = os.listdir(TEST_LOG_FOLDER)
    for test_log_file in test_log_files:
        reward, entropy = [], []
        with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
            for line in f:
                parse = line.split()
                try:
                    entropy.append(float(parse[-2]))
                    reward.append(float(parse[-1]))
                except IndexError:
                    break
        rewards_list.append(float(np.mean(reward[1:])))
        entropies_list.append(float(np.mean(entropy[1:])))
    rewards = np.array(rewards_list)
    rewards_min = np.min(rewards)
    rewards_5per = np.percentile(rewards, 5)
    rewards_mean = np.mean(rewards)
    rewards_median = np.percentile(rewards, 50)
    rewards_95per = np.percentile(rewards, 95)
    rewards_max = np.max(rewards)
    log_file.write(str(epoch) + '\t' +
                   str(rewards_min) + '\t' +
                   str(rewards_5per) + '\t' +
                   str(rewards_mean) + '\t' +
                   str(rewards_median) + '\t' +
                   str(rewards_95per) + '\t' +
                   str(rewards_max) + '\n')
    log_file.flush()
    return rewards_mean, np.mean(entropies_list)
        
def central_agent(net_params_queues, exp_queues):
    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    # TODO: restore neural net parameters
    actor = network.PPOAgent(BOARD_XSIZE, BOARD_YSIZE)

    # Get Writer
    writer = tf.summary.create_file_writer(SUMMARY_DIR);

    step=0

    summary_reward_buf:list[float] = []
    summary_entropy_buf:list[float] = []

    with writer.as_default():
        for epoch in range(TRAIN_EPOCHS):
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            for i in range(NUM_AGENTS):
                net_params_queues[i].put(actor_net_params)
            
            s_batch:list[env.Observation] = []
            a_batch:list[env.Action] = []
            p_batch:list[npt.NDArray[np.float32]]  = []
            d_batch:list[env.Advantage] = []
            v_batch:list[env.Value] = []
            for _ in range(BATCH_SIZE//NUM_AGENTS):
                for i in range(NUM_AGENTS):
                    s_, a_, p_, d_, v_  = exp_queues[i].get()
                    s_batch += s_
                    a_batch += a_
                    p_batch += p_
                    d_batch += d_
                    v_batch += v_

                    summary_reward_buf.append(float(v_[-1]))
                    summary_entropy_buf.append(actor.get_entropy_weight())

            steps_trained = actor.train(s_batch, a_batch, d_batch, v_batch, p_batch, step)

            step += steps_trained


            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Save the neural net parameters to disk.
                actor_path = f"{SUMMARY_DIR}/nn_model_ep_{epoch}_actor.ckpt"
                critic_path = f"{SUMMARY_DIR}/nn_model_ep_{epoch}_critic.ckpt"
                save_path = actor.save(actor_path, critic_path)

                # Write to Log File
                #avg_reward, avg_entropy = testing(epoch, actor_path, critic_path, test_log_file)
                avg_reward = sum(summary_reward_buf)/len(summary_reward_buf)
                avg_entropy = sum(summary_entropy_buf)/len(summary_entropy_buf)
                tf.summary.scalar('avg_reward', avg_reward, step=step)
                tf.summary.scalar('avg_entropy', avg_entropy, step=step)

                # clear
                summary_reward_buf = []
                summary_entropy_buf = []



def agent(agent_id, net_params_queue, exp_queue):
    e = env.Env(DIMS)
    actor = network.PPOAgent(BOARD_XSIZE, BOARD_YSIZE)

    # initial synchronization of the network parameters from the coordinator
    actor_net_params = net_params_queue.get()
    actor.set_network_params(actor_net_params)

    ACTOR_ID = np.int8(1)
    OPPONENT_ID = np.int8(2)

    for epoch in range(TRAIN_EPOCHS):
        e.reset()
        s_batch:list[env.Observation] = []
        a_batch:list[env.Action] = []
        p_batch:list[npt.NDArray[np.float32]]  = []
        r_batch:list[env.Reward] = []
        actor_turn = True
        while not e.game_over():
            if actor_turn:
                obs = e.observe(ACTOR_ID)

                action_prob = actor.predict_batch([obs])[0]
                print(action_prob)
                print(env.print_obs(obs))

                # gumbel noise
                noise = np.random.gumbel(size=len(action_prob))
                chosen_action: env.Action = np.argmax(np.log(action_prob) + noise)

                s_batch.append(obs)

                reward, obs = e.step(chosen_action, ACTOR_ID)
                print(env.print_obs(obs))


                a_batch.append(chosen_action)
                r_batch.append(reward)
                p_batch.append(action_prob)
            else:
                # random opponent for now
                opponent_action = np.int8(np.random.randint(0, BOARD_XSIZE))
                e.step(opponent_action, OPPONENT_ID)
            # flip turn
            actor_turn = not actor_turn

        v_batch = actor.compute_value(r_batch)
        d_batch = actor.compute_advantage(s_batch, r_batch)
        exp_queue.put([s_batch, a_batch, p_batch, d_batch, v_batch])

        actor_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)

def main():
    np.random.seed(RANDOM_SEED)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in range(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in range(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()


if __name__ == '__main__':
    main()
