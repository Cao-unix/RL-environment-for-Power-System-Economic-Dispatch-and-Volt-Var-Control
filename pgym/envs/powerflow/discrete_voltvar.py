from pgym.envs.powerflow.voltvar import VoltVarEnv
from pgym.pf_cases import case5bw
import numpy as np
from numpy import array
from pypower.idx_gen import QG, QMAX, QMIN
from pypower.idx_bus import BUS_I, BS
from gym import spaces


class DiscreteVoltVarEnv(VoltVarEnv):

    def __init__(self, case=None, **kwargs):
        if case is None:
            case = case5bw()
        info = {
            'case_name': 'dvvpfe',
            'sigma': 5,                 # discrete pieces
        }
        info.update(kwargs)
        self.sigma = info['sigma']
        super().__init__(case, **info)

    def get_action_space(self):
        cnt = len(self.case0['controlled_gen'][:, 0])
        min_action = np.array([self.case0['gen'][i, QMIN]
                               for i in self.case0['controlled_gen'][:, 0]])
        max_action = np.array([self.case0['gen'][i, QMAX]
                               for i in self.case0['controlled_gen'][:, 0]])
        self.action_scale = (max_action - min_action) / self.sigma
        return spaces.Discrete(cnt * 2), min_action, max_action

    def put_action(self, action):
        case = self.case
        k = action // 2
        sign = 1 - 2 * (action % 2)
        # print(sign)
        tov = case['gen'][case['controlled_gen']
                          [k, 0], QG] + sign * self.action_scale[k]
        tov = max([tov, self.min_action[k]])
        tov = min([tov, self.max_action[k]])
        case['gen'][case['controlled_gen'][k, 0], QG] = tov
        return case

# 增加电容器投切
class DiscreteVoltVarEnvCap(VoltVarEnv):
    """
    发电机Q离散调节 + 电容器多档位投切（0..cap_sigma）
    动作：对每个“受控发电机 + 电容器”，提供 (+) / (-) 两种动作
    """

    def __init__(self, case=None, **kwargs):
        if case is None:
            case = case5bw()

        info = {
            'case_name': 'dvvpfe_cap',
            'sigma': 5,        # 发电机Q离散档数
            'cap_sigma': 10,    # 电容器档数：0..cap_sigma
        }
        info.update(kwargs)
        self.sigma = int(info['sigma'])
        self.cap_sigma = int(info['cap_sigma'])

        # ===== 读取固定电容器配置（来自 case）=====
        self.cap_bus = np.array(case.get("cap_bus", []), dtype=int).reshape(-1)
        self.cap_q   = np.array(case.get("cap_q",   []), dtype=float).reshape(-1)
        assert len(self.cap_bus) == len(self.cap_q), "cap_bus 和 cap_q 长度必须一致"

        bus_ids = case["bus"][:, BUS_I].astype(int)
        busid2row = {bid: i for i, bid in enumerate(bus_ids)}
        self.cap_rows = np.array([busid2row[b] for b in self.cap_bus], dtype=int)
        self.n_cap = len(self.cap_rows)

        # 电容器当前档位：0..cap_sigma
        self.cap_state = np.zeros(self.n_cap, dtype=int)

        # 每一档对应的 MVAr
        self.cap_step = self.cap_q / float(self.cap_sigma) if self.n_cap > 0 else np.array([])

        super().__init__(case, **info)

        # OFF 基准 Bs（用于叠加）
        self.base_bs = self.case0["bus"][:, BS].copy()

    def get_action_space(self):
        cnt_gen = len(self.case0['controlled_gen'][:, 0])

        min_action = np.array([self.case0['gen'][i, QMIN]
                               for i in self.case0['controlled_gen'][:, 0]])
        max_action = np.array([self.case0['gen'][i, QMAX]
                               for i in self.case0['controlled_gen'][:, 0]])

        self.action_scale = (max_action - min_action) / self.sigma

        cnt_total = cnt_gen + self.n_cap
        return spaces.Discrete(cnt_total * 2), min_action, max_action

    def put_action(self, action):
        case = self.case
        cnt_gen = len(self.case0['controlled_gen'][:, 0])

        k = action // 2
        sign = 1 - 2 * (action % 2)   # 0 -> +1, 1 -> -1

        # ===== 发电机Q调节（原逻辑）=====
        if k < cnt_gen:
            tov = case['gen'][case['controlled_gen'][k, 0], QG] + sign * self.action_scale[k]
            tov = max([tov, self.min_action[k]])
            tov = min([tov, self.max_action[k]])
            case['gen'][case['controlled_gen'][k, 0], QG] = tov
            return case

        # ===== 电容器多档位投切 =====
        ci = k - cnt_gen
        self.cap_state[ci] = int(np.clip(self.cap_state[ci] + sign, 0, self.cap_sigma))

        r = self.cap_rows[ci]
        case['bus'][r, BS] = self.base_bs[r] + self.cap_state[ci] * self.cap_step[ci]
        return case

    # 新增电容器状态
    def get_observation(self, case=None):
        obs = super().get_observation(case)
        # cap_state: 0..cap_sigma
        if self.n_cap > 0:
            obs['cap'] = self.cap_state.astype(np.float32).copy()
        return obs

    def get_observation_space(self):
        obs_space, low, high = super().get_observation_space()

        if self.n_cap == 0:
            return obs_space, low, high

        cap_low = np.zeros(self.n_cap, dtype=np.float32)
        cap_high = np.ones(self.n_cap, dtype=np.float32) * float(self.cap_sigma)

        low2 = np.concatenate([low.astype(np.float32), cap_low])
        high2 = np.concatenate([high.astype(np.float32), cap_high])

        self.low_state, self.high_state = low2, high2
        self.observation_space = spaces.Box(low=low2, high=high2, dtype=np.float32)
        return self.observation_space, self.low_state, self.high_state

    def reset(self, absolute=False):
        # 先把电容器状态清零，保证 reset 后观测一致
        if self.n_cap > 0:
            self.cap_state[:] = 0
        return super().reset(absolute=absolute)

