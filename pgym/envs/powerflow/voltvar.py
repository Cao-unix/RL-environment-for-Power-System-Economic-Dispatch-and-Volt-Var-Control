from pgym.envs.powerflow.core import PowerFlowEnv, Observation
from pgym.utils.power import loss
import numpy as np
from numpy import array
from gym import spaces
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF, VMAX, VMIN
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS, PMAX, PMIN


class VoltVarEnv(PowerFlowEnv):
    def __init__(self, case, **kwargs):
        info = {
            'case_name': 'vvpfe',
            'reward_func': 'ploss_vr',     # reward function
            'VR': 100,                     # voltage penalty ratio
            'ObsT': False,
        }
        info.update(kwargs)
        self.reward_func = info['reward_func']
        self.VR = info['VR']
        self.ObsT = info['ObsT']
        super().__init__(case, **info)

    def get_observation_space(self):

        def get_obs_bound(case0):
            tl = 0
            vml = case0['bus'][:, VMIN]
            pdl = array([0 for p in case0['bus'][:, PD]])
            qdl = array([0 for q in case0['bus'][:, QD]])
            pgl = case0['gen'][:, PMIN]
            qgl = case0['gen'][:, QMIN]

            tm = self.T
            vmm = case0['bus'][:, VMIN]
            #! warning: hardcoded pdm, qdm
            pdm = array([p * 5 for p in case0['bus'][:, PD]])
            qdm = array([q * 5 for q in case0['bus'][:, QD]])
            pgm = case0['gen'][:, PMAX]
            qgm = case0['gen'][:, QMAX]

            if self.ObsT:
                return np.concatenate([[tl], vml, pdl, qdl, pgl, qgl]), np.concatenate([[tm], vmm, pdm, qdm, pgm, qgm])
            return np.concatenate([vml, pdl, qdl, pgl, qgl]), np.concatenate([vmm, pdm, qdm, pgm, qgm])

        self.low_state, self.high_state = get_obs_bound(self.case0)
        self.observation_space = spaces.Box(low=self.low_state.astype(np.float32),
                                            high=self.high_state.astype(np.float32),
                                            dtype=np.float32)
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

    def get_reward(self, last_obs, obs):
        indices = self.get_indices(obs)
        assert indices['pLoss'] >= 0, 'PLoss < 0, error!'
        if self.reward_func == 'ploss':
            return -indices['pLoss']
        if self.reward_func == 'ploss_v':
            return -indices['pLoss'] - self.VR * indices['vViol']
        if self.reward_func == 'ploss_vr':
            return -indices['pLoss'] - self.VR * indices['vViolRate']
        if self.reward_func == 'd_ploss':
            return -indices['pLoss'] + self.get_indices(last_obs)['pLoss']
        return 0
    
    def get_reward_from_results(self, results):
        indices = self.get_indices(self.get_observation(results))
        indices['pLoss'], _ = loss(results)
        assert indices['pLoss'] >= 0, 'PLoss < 0, error!'
        if self.reward_func == 'ploss':
            return -indices['pLoss']
        if self.reward_func == 'ploss_v':
            return -indices['pLoss'] - self.VR * indices['vViol']
        if self.reward_func == 'ploss_vr':
            return -indices['pLoss'] - self.VR * indices['vViolRate']
        return 0

    def get_indices(self, obs=None):
        if not obs:
            obs = self.get_observation()
        vmax = self.case0['bus'][:, VMAX]
        vmin = self.case0['bus'][:, VMIN]
        if 'vmax' in self.case0:
            vmax = vmax * 0 + self.case0['vmax']
        if 'vmin' in self.case0:
            vmin = vmin * 0 + self.case0['vmin']

        pLoss = sum(obs['pg']) - sum(obs['pd'])
        vViol = sum(obs['vm'] > vmax) + sum(obs['vm'] < vmin)
        vViolRate = sum(np.clip(obs['vm'] - vmax, 0, None)**2 + np.clip(-obs['vm'] + vmin, 0, None)**2)

        return {
            'pLoss': pLoss,
            'vViol': vViol,
            'vViolRate': vViolRate
        }
