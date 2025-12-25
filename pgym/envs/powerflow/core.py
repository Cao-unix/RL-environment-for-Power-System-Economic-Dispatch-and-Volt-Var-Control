# Power flow environment
from copy import deepcopy
import numpy as np
from numpy import array
import os.path as osp
import gym
from gym.utils import seeding
from pypower.api import runpf, ppoption
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PQ, REF, VMAX, VMIN
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS, PMAX, PMIN
import matplotlib.pyplot as plt
import time
from pgym.utils.power import interp_q

ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)


class Observation(dict):
    def numpy(self):
        return np.concatenate([v for v in self.values()])


class GlobalObservation():
    """
    Global Observation for multi-agent
    """    
    def __init__(self, obs, obss):
        """
        init

        Arguments:
            obs {Observation} -- all observations
            obss {list of Observation} -- local observations
        """        
        self.obs = obs
        self.obss = obss
    
    def numpy(self):
        return self.obs.numpy()
    
    def local(self, i):
        return self.obss[i].numpy()
    
    def locals(self):
        return [o.numpy() for o in self.obss]

class PowerFlowEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # Parameters:
    ## Name     Value               Default
    ### VR      volt punish ratio   None
    ### TVSpd   time-variant speed  None
    def __init__(self, case, **kwargs):
        info = {
            'T': 96,                    # period
            'TV': False,                # time-variant?
            'TVSpd': 1,
            'TStart': None,
            'failed_reward': -50,
            'case_name': 'pfe',         # case name
            'log_dir': 'data/',         # save image dir
        }
        info.update(kwargs)
        self.T = info['T']
        self.case_name = info['case_name']
        self.log_dir = info['log_dir']
        self.failed_reward = info['failed_reward']
        self.case0 = deepcopy(case)
        self.case = deepcopy(case)

        self.time = 0
        self.abs_time = 0 # absolute time
        self.ax = None
        self.rendered = False
        # time-variant data length
        self.TVDL = 0
        self.TVSpd = info['TVSpd']
        if 'load_p' in case.keys() and 'gen_p' in case.keys() and info['TV']:
            self.TVDL = case['load_p'].shape[1]
            self.TStart = info['TStart'] or 0
        else:
            self.TStart = info['TStart']

        self.action_space, self.min_action, self.max_action = self.get_action_space()
        self.observation_space, self.low_state, self.high_state = self.get_observation_space()

        self.seed()
        self.reset()

    # return self.observation_space, self.low_state, self.high_state
    def get_observation_space(self):
        raise NotImplementedError()

    # return Observation
    def get_observation(self, case=None):
        raise NotImplementedError()
    
    # return dict()
    def get_indices_from_obs(self, obs=None):
        raise NotImplementedError()

    # return spaces.Discrete(cnt * 2), low_ctrl_q, high_ctrl_q
    def get_action_space(self):
        raise NotImplementedError()

    # return case
    def put_action(self, action):
        raise NotImplementedError()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # return scalar
    def get_reward(self, last_obs, obs):
        raise NotImplementedError()
    
    def get_reward_from_results(self, results):
        raise NotImplementedError()

    def step(self, action):
        # manage time-variant variables
        self.last_obs = self.get_observation()
        # simulate next step
        if self.TVDL > 0:
            ## update case via time
            t = (self.time + self.TStart) / self.TVSpd % self.TVDL
            self.case['bus'][:, PD] = self.case0['bus'][:, PD] * interp_q(self.case['load_p'], t)
            self.case['gen'][:, PG] = self.case0['gen'][:, PG] * interp_q(self.case['gen_p'], t)
        # run action
        tmp_case = deepcopy(self.case)
        self.put_action(action)
        # reset to default origin
        self.case['bus'][:, VM] = 1
        self.case['bus'][:, VA] = 0
        # warm start
        # self.case['bus'][:, VM] = tmp_case['bus'][:, VM]
        # self.case['bus'][:, VA] = tmp_case['bus'][:, VA]

        results, self.success = runpf(self.case, ppopt)

        if not self.success:
            done = True
            # print(f'PF Error at {self.abs_time}')
            # import pickle
            # with open('data/result.pkl', 'wb') as file:
            #     pickle.dump(results, file)
            # assert False
            # ! if failed, use previous state and penal on reward
            results = tmp_case
            # get observation
            obs = self.get_observation(results)
            self.reward = self.failed_reward
        else:
            done = False
            # get observation
            obs = self.get_observation(results)
            # self.reward = self.get_reward(self.last_obs, obs)
            self.reward = self.get_reward_from_results(results)

        if (self.time + 1) % self.T == 0:
            done = True
        # next state
        self.case = results
        self.time += 1
        self.abs_time += 1
        ## manage time-variant variables
        return obs.numpy(), self.reward, done, {}

    def reset(self, absolute=False):
        # reset state
        self.time = 0
        self.case = deepcopy(self.case0)
        if absolute:
            self.abs_time = 0
        if self.TVDL > 0:
            t = (self.time + self.TStart) / self.TVSpd % self.TVDL
            self.case['bus'][:, PD] = self.case0['bus'][:, PD] * interp_q(self.case['load_p'], t)
            self.case['gen'][:, PG] = self.case0['gen'][:, PG] * interp_q(self.case['gen_p'], t)
        elif self.TStart is not None:
            t = self.TStart
            self.case['bus'][:, PD] = self.case0['bus'][:, PD] * interp_q(self.case['load_p'], t)
            self.case['gen'][:, PG] = self.case0['gen'][:, PG] * interp_q(self.case['gen_p'], t)
        results, self.success = runpf(self.case, ppopt)
        assert self.success, "Reset case unsolvable."
        self.case = results
        obs = self.get_observation()
        self.last_obs = obs
        return obs.numpy()

    # TODO
    def render(self, mode='human'):
        if mode == 'human.abs' or mode == 'human':
            self.rendered = True
            if self.ax is None:
                plt.show()
                self.tmax = 100
                self.rmax = -100
                self.rmin = 100
                self.ax = plt.gca()
                self.t_data = list()
                self.r_data = list()
                self.line, = self.ax.plot(self.t_data, self.r_data, '-')
                self.ax.set_title(f'{self.case_name} reward')
                self.ax.set_xlabel('Step')
                self.ax.set_ylabel('Reward')
                self.ax.set_xlim(0, self.tmax)
                self.ax.set_ylim(self.rmin, self.rmax)
            reward = self.reward
            if mode == 'human':
                t = self.time
                if t == 1:
                    self.t_data.clear()
                    self.r_data.clear()
            else:
                t = self.abs_time
            self.t_data.append(t)
            self.r_data.append(reward)
            self.line.set_xdata(self.t_data)
            self.line.set_ydata(self.r_data) 
            # adaptive drawing
            if (t > self.tmax):
                self.tmax = t * 2
            self.rmax = max([self.rmax, reward * 1.1, reward * 0.9])
            self.rmin = min([self.rmin, reward * 1.1, reward * 0.9])
            self.ax.set_xlim(0, self.tmax)
            self.ax.set_ylim(self.rmin, self.rmax)
            plt.draw()
            #! warning: delay
            plt.pause(1e-17)
        else:
            super().render(mode=mode) # just raise an exception

    def close(self):
        # close
        # print('saving & closing ...')
        if self.rendered:
            plt.savefig(osp.join(self.log_dir, f'{self.case_name}.png'))
            plt.savefig(osp.join(self.log_dir, f'{self.case_name}.svg'))
            plt.show()
