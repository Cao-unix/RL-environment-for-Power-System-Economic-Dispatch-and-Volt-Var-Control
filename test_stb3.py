from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import gymnasium as gym
import numpy as np
import gym as old_gym
from utils import make_env

def convert_space(space):
    if isinstance(space, old_gym.spaces.Box):
        return gym.spaces.Box(
            low=np.array(space.low, copy=True),
            high=np.array(space.high, copy=True),
            shape=space.shape,
            dtype=space.dtype,   # 保留原本 dtype（你这里是 float32）
        )
    if isinstance(space, old_gym.spaces.Discrete):
        return gym.spaces.Discrete(space.n)
    if isinstance(space, old_gym.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete(np.array(space.nvec, copy=True))
    if isinstance(space, old_gym.spaces.MultiBinary):
        return gym.spaces.MultiBinary(space.n)
    if isinstance(space, old_gym.spaces.Tuple):
        return gym.spaces.Tuple(tuple(convert_space(s) for s in space.spaces))
    if isinstance(space, old_gym.spaces.Dict):
        return gym.spaces.Dict({k: convert_space(v) for k, v in space.spaces.items()})
    raise TypeError(f"Unsupported space type: {type(space)}")

class GymToGymnasium(gym.Env):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.observation_space = convert_space(env.observation_space)
        self.action_space = convert_space(env.action_space)
        self.metadata = getattr(env, "metadata", {})

        # 记住原 env 的 action dtype（SB3 通常给 float32，但老 env 可能想要 float64）
        self._orig_action_dtype = getattr(getattr(env, "action_space", None), "dtype", None)

    def _cast_obs(self, obs):
        # 强制把 obs 转成 space 指定 dtype（你这里是 float32）
        return np.asarray(obs, dtype=self.observation_space.dtype)

    def reset(self, seed=None, options=None):
        if seed is not None:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
            if hasattr(self.env.action_space, "seed"):
                self.env.action_space.seed(seed)

        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
        else:
            obs, info = out, {}
        return self._cast_obs(obs), info

    def step(self, action):
        # 把 action 转回老 env 可能期望的 dtype（常见是 float64）
        if self._orig_action_dtype is not None:
            action = np.asarray(action, dtype=self._orig_action_dtype)

        out = self.env.step(action)
        if len(out) == 4:
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = False
        else:
            obs, reward, terminated, truncated, info = out

        return self._cast_obs(obs), float(reward), bool(terminated), bool(truncated), info

        
ENV_ID = "DContinuousEconomicDispatch-v0"
CASE = "case9"

# ---------------------------
# 1) 构造单个环境（给 DummyVecEnv 用）
# ---------------------------
def make_one():
    env = make_env(ENV_ID, CASE)[0]   # 你现在返回 (env, state_dim, action_dim)
    env = GymToGymnasium(env)         # 兼容 gymnasium + cast obs/action
    env = Monitor(env)
    return env

# ---------------------------
# 2) 训练并保存（reward 归一化）
# ---------------------------
venv = DummyVecEnv([make_one])
venv = VecNormalize(
    venv,
    norm_obs=True,
    norm_reward=True,   # 训练时归一化 reward
    clip_obs=10.0
)

model = PPO(
    "MlpPolicy",
    venv,
    verbose=1,
    tensorboard_log="./tb_sb3/",
    n_steps=2048,
    batch_size=64,
    learning_rate=3e-4,
    device="cpu",       # MLP 通常 CPU 更快更稳
    vf_coef=0.1,        # 可选：先压低 value loss 权重
    ent_coef=0.01,      # 可选：给点探索
    clip_range_vf=0.2   # 可选：value clipping 稳定
)

model.learn(total_timesteps=200_000)

model.save("ppo_model")
venv.save("vecnormalize.pkl")
venv.close()

# ---------------------------
# 3) 评估（用真实 reward）
#    obs 仍然按训练统计归一化，但 reward 不归一化
# ---------------------------
eval_env = DummyVecEnv([make_one])
eval_env = VecNormalize.load("vecnormalize.pkl", eval_env)

eval_env.training = False     # 固定统计量
eval_env.norm_reward = False  # ✅ 输出真实 reward（不归一化）

model = PPO.load("ppo_model", env=eval_env)

n_episodes = 10
ep_returns = []

obs = eval_env.reset()
ep_ret = 0.0
ep_cnt = 0

while ep_cnt < n_episodes:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = eval_env.step(action)

    # reward 现在就是“真实尺度”
    ep_ret += float(reward[0])  # DummyVecEnv 返回 shape (n_envs,)
    if done[0]:
        ep_returns.append(ep_ret)
        print(f"[EVAL] episode {ep_cnt+1}: return(real) = {ep_ret:.3f}")
        ep_ret = 0.0
        ep_cnt += 1
        obs = eval_env.reset()

print("[EVAL] mean(real return) =", np.mean(ep_returns))
print("[EVAL] std(real return)  =", np.std(ep_returns))

eval_env.close()
