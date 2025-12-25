# econ_dispatch_cont.py
from pgym.envs.powerflow.econ_dispatch import EconomicDispatchEnv
from pgym.pf_cases import case33bw   # 你可以换成别的 test case
import numpy as np
from gym import spaces
from pypower.idx_gen import PG, PMAX, PMIN


class ContinuousEconomicDispatchEnv(EconomicDispatchEnv):

    def __init__(self, case=None, **kwargs):
        if case is None:
            case = case33bw()
        info = {
            'case_name': 'cedopf'
        }
        info.update(kwargs)
        super().__init__(case, **info)

    def get_action_space(self):
        # 受控机组列表 case0['controlled_gen'][:,0]
        idx = self.case0['controlled_gen'][:, 0].astype(int)

        min_action = np.array([self.case0['gen'][i, PMIN] for i in idx])
        max_action = np.array([self.case0['gen'][i, PMAX] for i in idx])

        self.min_action = min_action
        self.max_action = max_action

        return spaces.Box(
            low=min_action.astype(np.float32),
            high=max_action.astype(np.float32)
        ), min_action, max_action

    def put_action(self, action):
        case = self.case
        idx = case['controlled_gen'][:, 0].astype(int)
        tov = np.clip(action, self.min_action, self.max_action)
        case['gen'][idx, PG] = tov
        return case


class DContinuousEconomicDispatchEnv(EconomicDispatchEnv):
    
    def __init__(self, case=None, **kwargs):
        if case is None:
            case = case33bw()
        info = {
            'case_name': 'cedopf'
        }
        info.update(kwargs)
        super().__init__(case, **info)
        self.ramp_rate = info.get("ramp_rate", 0.05)


    def get_action_space(self):
        idx = self.case0['controlled_gen'][:, 0].astype(int)
        n = len(idx)

        # 让 PPO 输出 tanh 动作天然匹配 [-1, 1]
        min_action = -np.ones(n, dtype=np.float32)
        max_action =  np.ones(n, dtype=np.float32)

        self.min_action = min_action
        self.max_action = max_action

        # === ramp 参数：每一步最多动多少（MW）===
        # 你可以在 kwargs 里传 ramp_rate，比如 0.05 表示 5% 的 (PMAX-PMIN)
        ramp_rate = getattr(self, "ramp_rate", None)
        if ramp_rate is None:
            self.ramp_rate = 0.01  # 默认 5%
        # 每台机组的 ramp（MW）
        self._idx = idx
        self._pmin = self.case0['gen'][idx, PMIN].copy()
        self._pmax = self.case0['gen'][idx, PMAX].copy()
        self._ramp = self.ramp_rate * (self._pmax - self._pmin)

        return spaces.Box(low=min_action, high=max_action, dtype=np.float32), min_action, max_action

    def put_action(self, action):
        case = self.case
        idx = case['controlled_gen'][:, 0].astype(int)

        # 当前 PG（用当前 case 的 gen，不要用 case0）
        pg_old = case['gen'][idx, PG].copy()

        # action ∈ [-1,1] -> ΔPg ∈ [-ramp, +ramp]
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        dpg = a * self._ramp

        pg_new = pg_old + dpg
        pg_new = np.clip(pg_new, self._pmin, self._pmax)

        case['gen'][idx, PG] = pg_new
        return case
