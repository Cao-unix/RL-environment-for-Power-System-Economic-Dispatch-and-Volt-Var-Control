import gym
from pypower.api import printpf
from pgym.pf_cases import case3
env = gym.make('pgym:DiscreteVoltVarPowerflow-v0', case=case3())
env.reset()
print(env.case)

action_space = env.action_space
print(action_space)
print(env.action_scale)
obs = env.get_observation(case3())
print(obs)
idxs = env.get_indices()
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

