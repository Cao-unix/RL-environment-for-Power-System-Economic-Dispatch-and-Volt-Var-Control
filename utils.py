import gym
import pgym
import numpy as np
import torch


# Generic replay buffer for standard gym tasks
class StandardBuffer(object):
    def __init__(self, state_dim, action_dim, batch_size, buffer_size, device):
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.buffer_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

    def add(self, state, action, next_state, reward, done, episode_done, episode_start):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(
            max(0, self.crt_size - self.buffer_size), self.crt_size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy",
                self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(
            f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(
            f"{save_folder}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(
            f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(
            f"{save_folder}_not_done.npy")[:self.crt_size]

        print(f"Replay Buffer loaded with {self.crt_size} elements.")


# def make_env(env_name):
#     env = gym.make(env_name)
#     state_dim = env.observation_space.shape[0]
#     return (
#         env,
#         state_dim,
#         env.action_space.n
#     )

import pgym
from pgym import pf_cases

from gym import spaces

def make_env(env_name, case_name=None):
    if case_name is not None:
        case_func = getattr(pgym.pf_cases, case_name)  # 字符串 -> 函数
        env = gym.make(env_name, case=case_func())
    else:
        env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, spaces.Discrete):
        # 离散动作，比如 DQN
        num_actions = env.action_space.n
    elif isinstance(env.action_space, spaces.Box):
        # 连续动作：动作维度
        num_actions = env.action_space.shape[0]
    else:
        raise NotImplementedError(f"Unsupported action space type: {type(env.action_space)}")

    return env, state_dim, num_actions

