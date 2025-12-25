# econ_dispatch.py
from pgym.envs.powerflow.core import PowerFlowEnv, Observation
import numpy as np
from numpy import array
from gym import spaces
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF, VMAX, VMIN
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS, PMAX, PMIN


class EconomicDispatchEnv(PowerFlowEnv):

    def __init__(self, case, **kwargs):
        info = {
            'case_name': 'edopf',
            'ObsT': False,
        }
        info.update(kwargs)

        self.ObsT = info['ObsT']

        # 初始化燃料成本系数
        self._init_gencost(case)

        super().__init__(case, **info)

    # -------------------------
    # gencost 初始化 & 计算
    # -------------------------
    def _init_gencost(self, case):

        gen = case['gen']
        n_gen = gen.shape[0]

        if 'gencost' in case:
            self.gencost = np.array(case['gencost'])
            # 期望格式: [2, startup, shutdown, n, c2, c1, c0]
            assert self.gencost.shape[0] == n_gen, "gencost 行数和机组数不一致"
        else:
            gencost = []
            for gi in range(n_gen):
                if gi == 0:
                    # 比如 slack 稍微贵一点
                    c2, c1, c0 = 0.4, 20.0, 0.0
                else:
                    # 其余机组稍微便宜一点
                    c2, c1, c0 = 0.2, 10.0, 0.0
                gencost.append([2, 0.0, 0.0, 3, c2, c1, c0])
            self.gencost = np.array(gencost)

    def _fuel_cost(self, Pg):
        c2 = self.gencost[:, 4]
        c1 = self.gencost[:, 5]
        c0 = self.gencost[:, 6]
        return float(np.sum(c2 * Pg**2 + c1 * Pg + c0))

    # -------------------------
    # Observation 定义
    # -------------------------
    def get_observation_space(self):

        def get_obs_bound(case0):
            tl = 0
            vml = case0['bus'][:, VMIN]
            pdl = array([0 for _ in case0['bus'][:, PD]])
            qdl = array([0 for _ in case0['bus'][:, QD]])
            pgl = case0['gen'][:, PMIN]
            qgl = case0['gen'][:, QMIN]

            tm = self.T
            vmm = case0['bus'][:, VMAX]
            #! warning: hardcoded pdm, qdm
            pdm = array([p * 5 for p in case0['bus'][:, PD]])
            qdm = array([q * 5 for q in case0['bus'][:, QD]])
            pgm = case0['gen'][:, PMAX]
            qgm = case0['gen'][:, QMAX]

            if self.ObsT:
                low = np.concatenate([[tl], vml, pdl, qdl, pgl, qgl])
                high = np.concatenate([[tm], vmm, pdm, qdm, pgm, qgm])
            else:
                low = np.concatenate([vml, pdl, qdl, pgl, qgl])
                high = np.concatenate([vmm, pdm, qdm, pgm, qgm])
            return low, high
        
        # 潮流计算可能返回float64, 强制转换为float32
        self.low_state, self.high_state = get_obs_bound(self.case0)
        self.observation_space = spaces.Box(
            low=self.low_state.astype(np.float32),
            high=self.high_state.astype(np.float32),
            dtype=np.float32
        )
        return self.observation_space, self.low_state, self.high_state

    def get_observation(self, case=None):
        if case is None:
            case = self.case
        obs = Observation()
        if self.ObsT:
            obs['t'] = array([self.time])
        obs['vm'] = case['bus'][:, VM].copy()
        obs['pd'] = case['bus'][:, PD].copy()
        obs['qd'] = case['bus'][:, QD].copy()
        obs['pg'] = case['gen'][:, PG].copy()
        obs['qg'] = case['gen'][:, QG].copy()
        return obs

    # -------------------------
    # 指标 & 奖励函数
    # -------------------------
    def get_indices(self, obs=None):

        if not obs:
            obs = self.get_observation()

        pg = obs['pg']
        fuel_cost = self._fuel_cost(pg)

        return {
            'fuel_cost': fuel_cost
        }

    def get_reward(self, last_obs, obs):
        indices = self.get_indices(obs)
        fc = indices['fuel_cost']
        return -fc

    def get_reward_from_results(self, results):
        indices = self.get_indices(self.get_observation(results))
        fc = indices['fuel_cost']
        return -fc
