# run this script. Copyright:Yaqi Sun
# visualization:tensorboard --logdir C:\Users\SYQ\Desktop\Homework_Platform\rl_example2\data\results
# attention the default value of the length of the episode is 96
import argparse
import copy
import os
import gym
import pgym
import numpy as np
from tensorboardX import SummaryWriter

import DDQN

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.reset_default_graph()

# ***********************************************************************************************************************************
def make_env(env_name):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    return (
        env,
        state_dim,
        env.action_space.n
    )


def interact_with_environment(env, num_actions, state_dim, args, parameters):
    # Initialize and load policy
    policy = DDQN.DDQN(
        num_actions,
        state_dim,
        parameters["learning_rate"],
        parameters["discount"],
        parameters["end_eps"],
        parameters["initial_eps"],
        parameters["eps_increment"],
        parameters["target_update_freq"],
        parameters["buffer_size"],
        parameters["batch_size"],
    )

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    pgym_pLoss = 0
    pgym_vViol = 0
    pgym_vViolRate = 0
    pgym_reward = 0
    train_sample = np.zeros((1, state_dim * 2 + 2))

    # Interact with the environment for max_timesteps
    '''
    Parameter interpretation:
        t:the number of the episode training time step
    '''
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        # if t<start_timestep,random choose;else choose according to the rule
        if t < parameters["start_timesteps"]:
            action = env.action_space.sample()
        else:
            action = policy.choose_action(np.array(state))

        # Perform action and log results
        next_state, reward, done, _ = env.step(action)
        transition = np.hstack((state, [action, reward], next_state))
        train_sample[0, :] = transition
        episode_reward += reward

        # Shallow copy,but it doesn't matter
        state = copy.copy(next_state)

        # Train agent after collecting sufficient data
        policy.learn(train_sample)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} \
                Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        ind = env.get_indices()
        pgym_pLoss += ind['pLoss']
        pgym_vViol += ind['vViol']
        pgym_vViolRate += ind['vViolRate']
        pgym_reward += reward
        if (t + 1) % 96 == 0:
            pgym_pLoss /= 96
            pgym_vViol /= 96
            pgym_vViolRate /= 96
            summary.add_scalar('pLoss', pgym_pLoss, t)
            summary.add_scalar('vViol', pgym_vViol, t)
            summary.add_scalar('vViolRate', pgym_vViolRate, t)
            summary.add_scalar('reward', pgym_reward, t)
            pgym_pLoss = 0
            pgym_vViol = 0
            pgym_vViolRate = 0
            pgym_reward = 0
# **********************************************************************************************************************************


if __name__ == "__main__":

    regular_parameters = {
        # Exploration
        "start_timesteps": 480,
        "initial_eps": 0.55,
        "end_eps": 0.95,
        "eps_increment": 0.00001,
        # Learning
        "learning_rate": 0.001,
        "discount": 0.9,
        "buffer_size": 48000,
        "batch_size": 256,
        "train_freq": 1,
        "target_update_freq": 32,
    }

    '''
    Argsparse is the standard module for Command-line parsing in Python, built into Python, and does not require installation.
    This library allows you to pass in parameters and run the program directly from the command line.
    '''
    # Load parameters
    parser = argparse.ArgumentParser()
    # OpenAI gym environment name
    parser.add_argument("--env", default="DiscreteVoltVarPowerflow-v0")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # Prepends name to filename
    parser.add_argument("--buffer_name", default="Default")
    # Max time steps to run environment or train for
    parser.add_argument("--max_timesteps", default=48000, type=int)
    # Max time steps to run environment or train for
    args = parser.parse_args()

    # Make folders
    if not os.path.exists("./data/results"):
        os.makedirs("./data/results")

    if not os.path.exists("./data/models"):
        os.makedirs("./data/models")

    if not os.path.exists("./data/buffers"):
        os.makedirs("./data/buffers")

    # Make env and determine properties
    env, state_dim, num_actions = make_env(args.env)
    parameters = regular_parameters
    '''
    When you update a dictionary a, there are two situations:
    (1) With the same key: the value corresponding to the key in the latest dictionary B will be used.
    (2) When there are new keys: the key and value in dictionary B will be added to A directly.
    '''

    # Record the training data
    summary = SummaryWriter('./data/results')
    interact_with_environment(
        env, num_actions, state_dim, args, parameters)
