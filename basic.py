import sys
sys.path.append(r"H:/caoshujun/homework/pgym/rl_example2")


import argparse
import copy
# import importlib
# import json
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

import DQN
import json
import utils
from datetime import datetime
summary = None


def interact_with_environment(env, replay_buffer, num_actions, state_dim, device, args, parameters):
    # For saving files
    setting = f"{args.env}_{args.seed}"

    # Initialize and load policy
    policy = DQN.DQN(
        num_actions,
        state_dim,
        device,
        parameters["discount"],
        parameters["optimizer"],
        parameters["optimizer_parameters"],
        parameters["polyak_target_update"],
        parameters["target_update_freq"],
        parameters["tau"],
        parameters["initial_eps"],
        parameters["end_eps"],
        parameters["eps_decay_period"],
        parameters["eval_eps"],
    )

    evaluations = []

    state, done = env.reset(), False
    episode_start = True
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    pgym_pLoss = 0
    pgym_vViol = 0
    pgym_vViolRate = 0
    pgym_reward = 0

    # Interact with the environment for max_timesteps
    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1
        if t < parameters["start_timesteps"]:
            action = env.action_space.sample()
        else:
            action = policy.select_action(np.array(state))

        # Perform action and log results
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        done_float = float(done)

        replay_buffer.add(state, action, next_state, reward,
                          done_float, done, episode_start)
        state = copy.copy(next_state)
        episode_start = False

        # Train agent after collecting sufficient data
        if t >= parameters["start_timesteps"] and (t+1) % parameters["train_freq"] == 0:
            info = policy.train(replay_buffer)
            for k,v in info.items():
                summary.add_scalar(k, v, t)

        if done:
            # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
            print(
                f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # Reset environment
            state, done = env.reset(), False
            episode_start = True
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

        # Evaluate episode
        if (t + 1) % parameters["eval_freq"] == 0:
            eval_reward = eval_policy(policy, args.env, args.seed,10)
            evaluations.append(eval_reward)
            np.save(f"./data/results/behavioral_{setting}", evaluations)
            policy.save(f"./data/models/behavioral_{setting}")

            # 把 evaluation 的平均回报写到 TensorBoard
            # tag 可以叫 eval/avg_reward，step 就用当前的 t+1
            summary.add_scalar("eval/avg_reward", eval_reward, t + 1)
            
    # Save final policy
    policy.save(f"./data/models/behavioral_{setting}")


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=1):
    eval_env, _, _ = utils.make_env(env_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            action = policy.select_action(np.array(state), eval=True)
            state, reward, done, _ = eval_env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    # default parameters
    # regular_parameters = {
    #     # Exploration
    #     "start_timesteps": 2,
    #     "initial_eps": 0.1,
    #     "end_eps": 0.1,
    #     "eps_decay_period": 1,
    #     # Evaluation
    #     "eval_freq": int(5e2),
    #     "eval_eps": 0,
    #     # Learning
    #     "discount": 0.99,
    #     "buffer_size": int(1e6),
    #     "batch_size": 4,
    #     "optimizer": "Adam",
    #     "optimizer_parameters": {
    #         "lr": 3e-2
    #     },
    #     "train_freq": 1,
    #     "polyak_target_update": True,
    #     "target_update_freq": 1,
    #     "tau": 0.005
    # }

    # adhust parameters
    regular_parameters = {
        # Exploration
        "start_timesteps": 2000,
        "initial_eps": 0.1,
        "end_eps": 0.01,
        "eps_decay_period": 3e4,
        # Evaluation
        "eval_freq": int(5e2),
        "eval_eps": 0,
        # Learning
        "discount": 0.95,
        "buffer_size": int(1e6),
        "batch_size": 64,
        "optimizer": "Adam",
        "optimizer_parameters": {
            "lr": 3e-4
        },
        "train_freq": 1,
        "polyak_target_update": False,
        "target_update_freq": 100,
        "tau": 0.005
    }


    # Load parameters
    parser = argparse.ArgumentParser()
    # OpenAI gym environment name
    parser.add_argument("--env", default="DiscreteVoltVarPowerflow-v0")
    # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--seed", default=0, type=int)
    # Prepends name to filename
    parser.add_argument("--buffer_name", default="Default")
    # Max time steps to run environment or train for
    parser.add_argument("--max_timesteps", default=1e5, type=int)
    parser.add_argument("--buffer_size", default=1e6, type=int)
    args = parser.parse_args()

    if not os.path.exists("./data/results"):
        os.makedirs("./data/results")

    if not os.path.exists("./data/models"):
        os.makedirs("./data/models")

    if not os.path.exists("./data/buffers"):
        os.makedirs("./data/buffers")

    # Make env and determine properties
    env, state_dim, num_actions = utils.make_env(args.env)
    parameters = regular_parameters
    parameters.update(vars(args))

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize buffer
    replay_buffer = utils.StandardBuffer(
        state_dim, 1, parameters["batch_size"], parameters["buffer_size"], device)



    #Tensorboard summary

    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M')}"
    logdir = os.path.join("./data/results/basic", run_name)
    summary = SummaryWriter(logdir)

    hparams = dict(parameters)

    opt_params = hparams.get("optimizer_parameters", {})
    for k, v in opt_params.items():
        hparams[f"opt_{k}"] = v

    hparams.pop("optimizer_parameters", None)

    hparams_str = json.dumps(hparams, indent=2, ensure_ascii=False)
    
    summary.add_text("config/hyperparameters", f"```json\n{hparams_str}\n```", 0)

    interact_with_environment(
        env, replay_buffer, num_actions, state_dim, device, args, parameters)
