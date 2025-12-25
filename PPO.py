import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class FC_Actor(nn.Module):
    """
    连续动作 Actor：输出 mu(s) + 全局可训练 log_std
    """
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


class FC_Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.v_head = nn.Linear(256, 1)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        return self.v_head(x)


def _gaussian_tanh_log_prob(dist: Normal, raw_action, eps=1e-6):
    logp_raw = dist.log_prob(raw_action).sum(-1, keepdim=True)  # [B,1]
    a = torch.tanh(raw_action)
    log_det = torch.log(1.0 - a * a + eps).sum(-1, keepdim=True)
    return logp_raw - log_det


def _scale_action(a_tanh, low, high):
    # [-1,1] -> [low,high]
    return low + (a_tanh + 1.0) * 0.5 * (high - low)


class PPO(object):
    """
    连续动作 PPO：内置 GAE
    - select_action(state, eval=False) -> (action_env, action_raw, logp, value)
    - train(buffer, last_state) -> dict stats
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        device,
        action_low,
        action_high,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        max_grad_norm=0.5,
        update_epochs=10,
        minibatch_size=256,
        target_kl=0.01,        # 可选：0.01 左右更稳
        value_clip=True,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl
        self.value_clip = value_clip

        self.state_shape = (-1, state_dim)

        self.action_low = torch.as_tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(action_high, dtype=torch.float32, device=device)

        self.actor = FC_Actor(state_dim, action_dim).to(device)
        self.critic = FC_Critic(state_dim).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )

        self.iterations = 0
        
    @torch.no_grad()
    def compute_gae_np(self, rewards, dones, values, last_value):
        rewards = np.asarray(rewards, dtype=np.float32)
        dones   = np.asarray(dones,   dtype=np.float32)
        values  = np.asarray(values,  dtype=np.float32)

        values_ext = np.append(values, np.float32(last_value))

        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values_ext[t + 1] * nonterminal - values_ext[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            adv[t] = gae

        ret = adv + values
        return adv, ret

    @torch.no_grad()
    def build_rollout(self, states_buf, actions_raw_buf, log_probs_buf,
                      rewards_buf, dones_buf, values_buf, last_state):

        s_tensor = torch.FloatTensor(np.asarray(last_state)).to(self.device).unsqueeze(0)
        last_value = float(self.critic(s_tensor).cpu().numpy()[0, 0])

        adv, ret = self.compute_gae_np(
            rewards=rewards_buf,
            dones=dones_buf,
            values=values_buf,
            last_value=last_value
        )

        states_tensor = torch.FloatTensor(np.asarray(states_buf)).to(self.device)
        actions_raw_tensor = torch.FloatTensor(np.asarray(actions_raw_buf)).to(self.device)
        log_probs_tensor = torch.FloatTensor(np.asarray(log_probs_buf)).view(-1, 1).to(self.device)

        # 采样时 critic 的 old values（用于 value clipping）
        values_tensor = torch.FloatTensor(np.asarray(values_buf, dtype=np.float32)).view(-1, 1).to(self.device)

        returns_tensor = torch.FloatTensor(ret).view(-1, 1).to(self.device)
        adv_tensor = torch.FloatTensor(adv).view(-1, 1).to(self.device)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        return {
            "states": states_tensor,
            "actions_raw": actions_raw_tensor,
            "log_probs": log_probs_tensor,
            "values": values_tensor,          # ✅ 新增
            "returns": returns_tensor,
            "advantages": adv_tensor,
        }


    @torch.no_grad()
    def select_action(self, state, eval=False):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(self.state_shape)

        mu, std = self.actor(s)
        dist = Normal(mu, std)

        raw_action = mu if eval else dist.rsample()                       
        logp = _gaussian_tanh_log_prob(dist, raw_action)                  

        a_tanh = torch.tanh(raw_action)
        action_env = _scale_action(a_tanh, self.action_low, self.action_high)

        v = self.critic(s)

        return (
            action_env.squeeze(0).detach().cpu().numpy(),
            raw_action.squeeze(0).detach(),   
            logp.squeeze(0).detach(),         
            v.squeeze(0).detach(),           
        )

    def evaluate(self, states, actions_raw):
        mu, std = self.actor(states)
        dist = Normal(mu, std)

        logp = _gaussian_tanh_log_prob(dist, actions_raw)               
        entropy = dist.entropy().sum(-1, keepdim=True)                  
        v = self.critic(states)                                           
        return logp, entropy, v

    @torch.no_grad()
    def compute_gae(self, rewards, terminated, values, last_value):
        """
        rewards:    [T,1]
        terminated: [T,1]  (terminated=1 -> mask=0)
        values:     [T,1]
        last_value: [1,1]
        """
        T = rewards.size(0)
        adv = torch.zeros_like(rewards)
        gae = torch.zeros((1, 1), device=rewards.device, dtype=rewards.dtype)

        masks = 1.0 - terminated

        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else values[t + 1]
            delta = rewards[t] + self.gamma * next_v * masks[t] - values[t]
            gae = delta + self.gamma * self.gae_lambda * masks[t] * gae
            adv[t] = gae.squeeze(0)

        returns = adv + values
        return adv, returns

    def train(self, rollout):
        states = rollout["states"].to(self.device)
        actions_raw = rollout["actions_raw"].to(self.device)
        old_logp = rollout["log_probs"].to(self.device)
        returns = rollout["returns"].to(self.device)
        adv = rollout["advantages"].to(self.device)
        old_values = rollout["values"].to(self.device)

        N = states.size(0)
        inds = np.arange(N)

        last_policy_loss = 0.0
        last_value_loss = 0.0
        last_entropy = 0.0
        last_kl = 0.0
        last_clipfrac = 0.0

        for _ in range(self.update_epochs):
            np.random.shuffle(inds)

            for start in range(0, N, self.minibatch_size):
                mb_idx = torch.as_tensor(
                    inds[start:start + self.minibatch_size],
                    dtype=torch.long,
                    device=self.device
                )

                mb_s = states[mb_idx]
                mb_a_raw = actions_raw[mb_idx]
                mb_old_logp = old_logp[mb_idx]
                mb_adv = adv[mb_idx]
                mb_ret = returns[mb_idx]

                new_logp, entropy, new_v = self.evaluate(mb_s, mb_a_raw)

                log_ratio = new_logp - mb_old_logp
                ratio = torch.exp(log_ratio)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

               
                mb_old_v = old_values[mb_idx]   # 采样时的 old V(s)

                if self.value_clip:
                    v_clipped = mb_old_v + torch.clamp(new_v - mb_old_v, -self.clip_eps, self.clip_eps)
                    v_loss1 = (new_v - mb_ret).pow(2)
                    v_loss2 = (v_clipped - mb_ret).pow(2)
                    value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(new_v, mb_ret)


                ent_mean = entropy.mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # 防止 log_std 过大过小
                self.actor.log_std.data.clamp_(-5.0, 2.0)


                with torch.no_grad():
                    approx_kl = (mb_old_logp - new_logp).mean().item()
                    clipfrac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean().item()

                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()
                last_entropy = ent_mean.item()
                last_kl = approx_kl
                last_clipfrac = clipfrac

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break
            if self.target_kl is not None and last_kl > self.target_kl:
                break

        self.iterations += 1

        return {
            "policy_loss": last_policy_loss,
            "value_loss": last_value_loss,
            "entropy": last_entropy,
            "approx_kl": last_kl,
            "clipfrac": last_clipfrac,
        }


    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor", map_location=self.device))
        self.critic.load_state_dict(torch.load(filename + "_critic", map_location=self.device))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer", map_location=self.device))
