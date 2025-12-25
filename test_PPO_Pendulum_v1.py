import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# 引入你的 PPO 类
import PPO 

# 创建目录
os.makedirs("./validation_logs", exist_ok=True)
os.makedirs("./validation_videos", exist_ok=True)

def make_gym_env(env_name):
    """创建一个标准的 Gym 环境"""
    # 尝试使用 render_mode (新版 Gym 特性)
    try:
        env = gym.make(env_name, render_mode="rgb_array")
    except:
        env = gym.make(env_name)
        
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high
    return env, state_dim, action_dim, action_low, action_high

def eval_policy(policy, env_name, seed, eval_episodes=5, record_video=False, run_name="test"):
    """
    评估策略
    """
    if record_video:
        # 新版 Gym 录像通常需要 wrapper 或者 render_mode，简单起见这里只做评估交互
        # 如果需要录像，建议单独处理，避免版本兼容问题
        pass
    
    env = gym.make(env_name)
    # 新版 Gym 没有 env.seed()，种子在 reset 时设置
    
    avg_reward = 0.0
    
    for i in range(eval_episodes):
        # 兼容新旧版本的 reset
        res = env.reset(seed=seed + 100 + i)
        if isinstance(res, tuple):
            state, _ = res  # 新版返回 (state, info)
        else:
            state = res     # 旧版返回 state
            
        done = False
        while not done:
            action, _, _, _ = policy.select_action(np.array(state), eval=True)
            
            # 兼容新旧版本的 step
            step_res = env.step(action)
            if len(step_res) == 5:
                # 新版: state, reward, terminated, truncated, info
                state, reward, terminated, truncated, _ = step_res
                done = terminated or truncated
            else:
                # 旧版: state, reward, done, info
                state, reward, done, _ = step_res
            
            avg_reward += reward
    
    env.close()
    return avg_reward / eval_episodes

def run_validation(args):
    # 1. 设置环境
    env_name = "Pendulum-v1" 
    env, state_dim, action_dim, action_low, action_high = make_gym_env(env_name)
    
    # 设置 PyTorch 和 NumPy 种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # 注意：env.seed(args.seed) 已被删除，将在 env.reset() 中设置
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Env: {env_name}, State Dim: {state_dim}, Action Dim: {action_dim}")

    # 2. 初始化你的 PPO
    policy = PPO.PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        action_low=action_low,
        action_high=action_high,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        update_epochs=10
    )

    # Tensorboard
    run_name = f"valid_{env_name}_{datetime.now().strftime('%Y%m%d-%H%M')}"
    writer = SummaryWriter(f"./validation_logs/{run_name}")

    # 3. 训练循环
    total_steps = 0
    max_steps = 200000 
    rollout_steps = 2048

    states_buf, actions_buf, logp_buf, rewards_buf, dones_buf, vals_buf = [], [], [], [], [], []
    
    # === 修改点：新版 Gym reset 用法 ===
    res = env.reset(seed=args.seed)
    if isinstance(res, tuple):
        state, _ = res
    else:
        state = res
        
    done = False
    episode_reward = 0
    
    log_rewards = []
    log_steps = []

    print("--- Start Training ---")

    while total_steps < max_steps:
        # 选择动作
        action_env, action_raw, logp, v = policy.select_action(state, eval=False)
        
        # === 修改点：新版 Gym step 用法 ===
        step_res = env.step(action_env)
        
        if len(step_res) == 5:
            # 新版 API
            next_state, reward, terminated, truncated, _ = step_res
            done = terminated or truncated
        else:
            # 旧版 API
            next_state, reward, done, _ = step_res
        
        # 存储
        states_buf.append(state)
        actions_buf.append(action_raw.cpu().numpy())
        logp_buf.append(float(logp.item()))
        rewards_buf.append(reward)
        dones_buf.append(float(done))
        vals_buf.append(float(v.item()))

        state = next_state
        episode_reward += reward
        total_steps += 1

        # Episode 结束
        if done:
            writer.add_scalar("Train/Episode_Reward", episode_reward, total_steps)
            print(f"Step: {total_steps}, Reward: {episode_reward:.2f}")
            
            # Reset
            res = env.reset() # 只有第一次需要 seed
            if isinstance(res, tuple):
                state, _ = res
            else:
                state = res
            
            done = False
            episode_reward = 0

        # PPO 更新
        if len(states_buf) >= rollout_steps:
            rollout = policy.build_rollout(
                states_buf, actions_buf, logp_buf, rewards_buf, dones_buf, vals_buf, state
            )
            train_info = policy.train(rollout)
            
            for k, v in train_info.items():
                writer.add_scalar(f"Train/{k}", v, total_steps)

            states_buf.clear()
            actions_buf.clear()
            logp_buf.clear()
            rewards_buf.clear()
            dones_buf.clear()
            vals_buf.clear()

        # 评估
        if total_steps % 10000 == 0:
            eval_r = eval_policy(policy, env_name, args.seed, record_video=False)
            writer.add_scalar("Eval/Reward", eval_r, total_steps)
            print(f">>> EVAL at {total_steps}: Avg Reward = {eval_r:.2f}")
            
            log_rewards.append(eval_r)
            log_steps.append(total_steps)

    env.close()
    
    # 画图
    plt.figure(figsize=(10, 5))
    plt.plot(log_steps, log_rewards)
    plt.title(f"PPO Validation on {env_name}")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.grid(True)
    plt.savefig(f"./validation_logs/{run_name}/result_plot.png")
    print(f"Training finished. Logs saved to ./validation_logs/{run_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    
    run_validation(args)