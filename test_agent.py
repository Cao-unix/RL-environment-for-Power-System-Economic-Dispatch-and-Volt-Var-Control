import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import utils
from pypower.idx_gen import GEN_BUS 
import matplotlib.pyplot as plt

def build_obs_names(env):
    e = env.unwrapped if hasattr(env, "unwrapped") else env
    case0 = getattr(e, "case0", None)
    if case0 is None:
        case0 = getattr(e, "case", None)
    if case0 is None:
        raise RuntimeError("拿不到 env.case0 / env.case，无法重命名状态维度")

    bus_ids = case0["bus"][:, 0].astype(int)          # bus_i
    gen_bus = case0["gen"][:, GEN_BUS].astype(int)    # each generator connected bus
    n_bus = len(bus_ids)
    n_gen = len(gen_bus)

    names = []
    # 兼容 ObsT
    if getattr(e, "ObsT", False):
        names.append("t")

    names += [f"vm_bus{b}" for b in bus_ids]
    names += [f"pd_bus{b}" for b in bus_ids]
    names += [f"qd_bus{b}" for b in bus_ids]
    names += [f"pg_gen{i+1}_bus{gen_bus[i]}" for i in range(n_gen)]
    names += [f"qg_gen{i+1}_bus{gen_bus[i]}" for i in range(n_gen)]
    return names, n_bus, n_gen


def rename_state_columns(df, env):
    obs_names, _, _ = build_obs_names(env)
    rename_map = {}
    for i, name in enumerate(obs_names):
        col = f"s{i}"
        if col in df.columns:
            rename_map[col] = name
    return df.rename(columns=rename_map)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        mu = self.mu_head(x)
        std = self.log_std.exp().expand_as(mu)
        return mu, std


def _extract_state_dict(obj: dict):
    """
    兼容多种保存格式，抽出 actor 的 state_dict
    """
    # 常见：{"actor": actor.state_dict(), ...}
    for k in ["actor", "actor_state_dict", "policy", "pi", "model", "state_dict"]:
        if k in obj and isinstance(obj[k], dict):
            return obj[k]

    if any(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj

    raise RuntimeError(f"无法从checkpoint里识别 actor state_dict，keys={list(obj.keys())[:20]}")


def load_actor(model_dir, device, state_dim, action_dim):
    actor_name = "behavioral_DContinuousEconomicDispatch-v0_0_20251214-0116_actor"
    actor_path = os.path.join(model_dir, actor_name)
    if not os.path.exists(actor_path):
        raise FileNotFoundError(f"找不到 actor 文件: {actor_path}")

    obj = torch.load(actor_path, map_location=device)

    actor = Actor(state_dim=state_dim, action_dim=action_dim).to(device)

    # 兼容不同保存格式
    if isinstance(obj, dict):
        sd = obj.get("actor", obj.get("actor_state_dict", obj))
        actor.load_state_dict(sd, strict=True)
    elif isinstance(obj, nn.Module):
        actor = obj.to(device)
    else:
        raise RuntimeError(f"不支持的actor格式: {type(obj)}")

    actor.eval()
    return actor



@torch.no_grad()
def actor_action(actor, obs, device, action_low=None, action_high=None, deterministic=True):
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    mu, std = actor(obs_t)

    if deterministic:
        act = mu
    else:
        act = mu + std * torch.randn_like(mu)

    act = act.squeeze(0).cpu().numpy()

    if action_low is not None and action_high is not None:
        act = np.clip(act, action_low, action_high)

    return act


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = "./data/models/optional2"

    env, state_dim, action_dim = utils.make_env("DContinuousEconomicDispatch-v0", "case9")
    actor = load_actor(model_dir, device, state_dim, action_dim)

    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out


    action_low = getattr(env.action_space, "low", None)
    action_high = getattr(env.action_space, "high", None)

    rows = []
    t = 0
    done = False

    while (not done) and t < 1000:
        act = actor_action(actor, obs, device, action_low, action_high, deterministic=True)

        step_out = env.step(act)

        # 兼容 gym (4) / gymnasium (5)
        if len(step_out) == 4:
            next_obs, reward, done, info = step_out
        else:
            next_obs, reward, terminated, truncated, info = step_out
            done = terminated or truncated

        row = {"t": t, "reward": float(reward)}
        for i, v in enumerate(np.array(obs).ravel()):
            row[f"s{i}"] = float(v)
        for i, v in enumerate(np.array(act).ravel()):
            row[f"a{i}"] = float(v)
        rows.append(row)

        obs = next_obs
        t += 1

    df = pd.DataFrame(rows)
    df = rename_state_columns(df, env)

    pg_cols = [c for c in df.columns if c.startswith("pg_gen")]
    df = df.iloc[:10]


    fig, ax1 = plt.subplots()

    # 左轴：Pg 曲线
    for c in pg_cols:
        ax1.plot(df["t"], df[c], label=c)
    ax1.set_xlabel("t")
    ax1.set_ylabel("Pg (MW)")
    ax1.set_title("Pg (left) and Reward (right) over First 10 Steps")

    # 右轴：reward
    ax2 = ax1.twinx()
    ax2.plot(df["t"], df["reward"], label="reward")
    ax2.set_ylabel("reward")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    fig.savefig("pg_reward_first10.png", dpi=200)
    print("Saved -> pg_reward_first10.png")



if __name__ == "__main__":
    main()
