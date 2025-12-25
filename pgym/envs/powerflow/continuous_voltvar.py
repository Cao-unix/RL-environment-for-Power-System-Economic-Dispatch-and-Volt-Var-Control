from pgym.envs.powerflow.voltvar import VoltVarEnv
from pgym.pf_cases import case39
import numpy as np
from numpy import array
from pypower.idx_gen import QG, QMAX, QMIN
from gym import spaces

class ContinuousVoltVarEnv(VoltVarEnv):

    def __init__(self, case=None, **kwargs):
        if case is None:
            case = case39()
        info = {
            'case_name': 'cvvpfe'
        }
        info.update(kwargs)
        super().__init__(case, **info)

    def get_action_space(self):
        min_action = np.array([self.case0['gen'][i, QMIN]
                        for i in self.case0['controlled_gen'][:, 0]])
        max_action = np.array([self.case0['gen'][i, QMAX]
                        for i in self.case0['controlled_gen'][:, 0]])
        return spaces.Box(min_action.astype(np.float32), max_action.astype(np.float32)), min_action, max_action

    def put_action(self, action):
        case = self.case
        tov = np.clip(action, self.min_action, self.max_action)
        case['gen'][case['controlled_gen'][:, 0], QG] = tov
        return case

class DContinuousVoltVarEnv(VoltVarEnv):

    def __init__(self, case=None, **kwargs):
        if case is None:
            case = case39()
        info = {
            'case_name': 'd_cvvpfe',
            'sigma': 8,                 # discrete pieces
        }
        info.update(kwargs)
        self.sigma = info['sigma']
        super().__init__(case, **info)

    def get_action_space(self):
        self.low_action = np.array([self.case0['gen'][i, QMIN]
                        for i in self.case0['controlled_gen'][:, 0]])
        self.high_action = np.array([self.case0['gen'][i, QMAX]
                        for i in self.case0['controlled_gen'][:, 0]])
        max_action = (self.high_action - self.low_action) / self.sigma
        min_action = -(self.high_action - self.low_action) / self.sigma
        return spaces.Box(min_action.astype(np.float32), max_action.astype(np.float32)), min_action, max_action

    def put_action(self, action):
        case = self.case
        tov = np.clip(action, self.min_action, self.max_action)
        tov += case['gen'][case['controlled_gen'][:, 0], QG]
        tov = np.clip(tov, self.low_action, self.high_action)
        case['gen'][case['controlled_gen'][:, 0], QG] = tov
        return case
