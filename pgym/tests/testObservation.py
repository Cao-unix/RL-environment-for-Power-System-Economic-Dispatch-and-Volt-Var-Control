from pgym.envs.powerflow.core import Observation

obs = Observation()
obs['vm'] = [2]
obs['pd'] = [3]
obs['qd'] = [-1]
obs['pg'] = [-9]
obs['qg'] = [5]

print(obs)
print([v for v in obs.values()])

obs = Observation({
    'vm': [2],
    'pd': [3],
    'qd': [-1],
    'pg': [-9],
    'qg': [5]
})

print(obs)
print([v for v in obs.values()])
