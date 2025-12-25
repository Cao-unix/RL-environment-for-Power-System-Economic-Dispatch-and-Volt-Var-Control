import gym
import time

ag = gym.make("pgym:DiscreteVoltVarPowerflow-v0")
action_spec = ag.action_space
ag.reset()
pa = action_spec.sample()
print(ag.step(pa))

ag.reset()
for i in range(100):
    pa = action_spec.sample()
    ag.step(pa)
    ag.render()
    time.sleep(1e-3)

ag.close()
