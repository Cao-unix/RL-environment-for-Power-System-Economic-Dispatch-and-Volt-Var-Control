import gym
from pypower.api import printpf
from pgym.pf_cases import case33bw
env = gym.make('pgym:ContinuousVoltVarPowerflow-v0', case=case33bw())
env.reset()
print(env.case)

action_space = env.action_space
print(action_space)
obs = env.get_observation()
print(obs)
idxs = env.get_indices()
print(idxs)

from pypower.api import runpf, ppoption, printpf
ppopt = ppoption(PF_ALG=1, VERBOSE=0, OUT_ALL=0)
results, success = runpf(case33bw(), ppopt)
obs = env.get_observation(results)
print(obs)
idxs = env.get_indices(obs)
print(idxs)

for i in range(2):
    env.reset()
    for j in range(96):
        action = action_space.sample()
        # print(action)
        obs, reward, done, others = env.step(action)
        env.render('human.abs')

obs = env.get_observation()
print(obs)
idxs = env.get_indices()
print(idxs)
env.close()

