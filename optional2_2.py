import argparse
import os
import json
from datetime import datetime

import numpy as np
import torch
from tensorboardX import SummaryWriter

import PPO
import utils

summary = None

# 修改点1：增加 run_name 参数，用于接收时间戳
def interact_with_environment(env, num_actions, state_dim, device, args, parameters, run_name):
    # For saving files
    setting = f"{args.env}_{args.seed}"

    # 连续动作维度
    action_dim = num_actions
    action_low = env.action_space.low
    action_high = env.action_space.high

    # Initialize PPO policy
    policy = PPO.PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        action_low=action_low,
        action_high=action_high,
        lr=parameters["optimizer_parameters"]["lr"],
        gamma=parameters["discount"],
        gae_lambda=parameters.get("gae_lambda", 0.95),
        clip_eps=parameters.get("clip_eps", 0.2),
        value_coef=parameters.get("value_coef", 0.5),
        entropy_coef=parameters.get("entropy_coef", 0.0),
        max_grad_norm=parameters.get("max_grad_norm", 0.5),
        update_epochs=parameters.get("update_epochs", 10),
    )

    evaluations = []

    state, done = env.reset(), False
    episode_reward = 0.0
    episode_timesteps = 0
    episode_num = 0

    total_steps = 0
    rollout_steps = int(parameters.get("rollout_steps", 2048))

    # on-policy rollout buffer
    states_buf = []
    actions_raw_buf = []
    log_probs_buf = []
    rewards_buf = []
    dones_buf = []
    values_buf = []

    pgym_reward = 0.0

    max_timesteps = int(args.max_timesteps)

    while total_steps < max_timesteps:
        # 选动作（训练时 eval=False）
        action_env, action_raw, logp, v = policy.select_action(
            np.array(state), eval=False
        )

        # 与环境交互
        next_state, reward, done, _ = env.step(action_env)

        episode_reward += reward
        episode_timesteps += 1
        total_steps += 1

        # 存到 on-policy rollout buffer
        states_buf.append(state)
        actions_raw_buf.append(action_raw.cpu().numpy())
        log_probs_buf.append(float(logp.item()))
        rewards_buf.append(reward)
        dones_buf.append(float(done))
        values_buf.append(float(v.item()))

        state = next_state
        pgym_reward += reward

        if done:
            print(
                f"Total T: {total_steps} "
                f"Episode Num: {episode_num + 1} "
                f"Episode T: {episode_timesteps} "
                f"Reward: {episode_reward:.3f}"
            )
            state, done = env.reset(), False
            episode_reward = 0.0
            episode_timesteps = 0
            episode_num += 1

        if total_steps % 96 == 0:
            avg_reward_96 = pgym_reward / 96.0
            summary.add_scalar("reward", avg_reward_96, total_steps)
            pgym_reward = 0.0

        if len(states_buf) >= rollout_steps or total_steps >= max_timesteps:
            rollout = policy.build_rollout(
                states_buf=states_buf,
                actions_raw_buf=actions_raw_buf,
                log_probs_buf=log_probs_buf,
                rewards_buf=rewards_buf,
                dones_buf=dones_buf,
                values_buf=values_buf,
                last_state=state,   # 用当前 state 做 bootstrap
            )

            info = policy.train(rollout)

            for k, v in info.items():
                summary.add_scalar(k, v, total_steps)

            states_buf.clear()
            actions_raw_buf.clear()
            log_probs_buf.clear()
            rewards_buf.clear()
            dones_buf.clear()
            values_buf.clear()

            
        # === Evaluation ===
        if total_steps % parameters["eval_freq"] == 0:
            eval_reward = eval_policy(
                policy, args.env, args.case, args.seed, eval_episodes=1
            )
            evaluations.append(eval_reward)

            np.save(f"./data/results/behavioral_{setting}", evaluations)
            policy.save(f"./data/models/behavioral_{setting}")

            summary.add_scalar("eval/avg_reward", eval_reward, total_steps)

    # Save final policy
    # 修改：保存到 optional2 子文件夹
    save_path = f"./data/models/optional2/behavioral_{setting}_{run_name}"
    if not os.path.exists("./data/models/optional2"):
        os.makedirs("./data/models/optional2")
    
    print(f"Saving final model to: {save_path}")
    policy.save(save_path)


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, case_name, seed, eval_episodes=1):
    eval_env, _, _ = utils.make_env(env_name, case_name)
    eval_env.seed(seed + 100)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        while not done:
            # 评估时 eval=True
            action_env, _, _, _ = policy.select_action(
                np.array(state), eval=True
            )
            state, reward, done, _ = eval_env.step(action_env)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    # PPO 的默认参数
    regular_parameters = {
        "start_timesteps": 0,
        "initial_eps": 0.0,
        "end_eps": 0.0,
        "eps_decay_period": 1,

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
        "tau": 0.005,

        # PPO 额外参数
        "rollout_steps": 1024,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.5,
        "max_grad_norm": 0.5,
        "update_epochs": 5,
    }

    # Load parameters
    parser = argparse.ArgumentParser()
    # OpenAI gym environment name
    parser.add_argument("--env", default="DContinuousEconomicDispatch-v0")
    # Case name
    parser.add_argument("--case", default="case9")
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
    env, state_dim, num_actions = utils.make_env(args.env, args.case)
    parameters = regular_parameters
    parameters.update(vars(args))

    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tensorboard summary
    run_name = f"{datetime.now().strftime('%Y%m%d-%H%M')}"
    logdir = os.path.join("./data/results/optional2", run_name)
    summary = SummaryWriter(logdir)

    hparams = dict(parameters)  

    opt_params = hparams.get("optimizer_parameters", {})
    for k, v in opt_params.items():
        hparams[f"opt_{k}"] = v
    hparams.pop("optimizer_parameters", None)

    hparams_str = json.dumps(hparams, indent=2, ensure_ascii=False)
    summary.add_text("config/hyperparameters",
                     f"```json\n{hparams_str}\n```", 0)

    interact_with_environment(
        env, num_actions, state_dim, device, args, parameters, run_name
    )