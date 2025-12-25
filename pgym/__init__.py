import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='DiscreteVoltVarPowerflow-v0',
    entry_point='pgym.envs:DiscreteVoltVarEnv',
)

register(
    id='ContinuousVoltVarPowerflow-v0',
    entry_point='pgym.envs:ContinuousVoltVarEnv',
)

register(
    id='DContinuousVoltVarPowerflow-v0',
    entry_point='pgym.envs:DContinuousVoltVarEnv',
)

# optional2 经济调度
register(
    id='ContinuousEconomicDispatch-v0',
    entry_point='pgym.envs:ContinuousEconomicDispatchEnv',
)

register(
    id='DContinuousEconomicDispatch-v0',
    entry_point='pgym.envs:DContinuousEconomicDispatchEnv',
)

# optional3 投切电容器
register(
    id='DiscreteVoltVarPowerflow-v1',
    entry_point='pgym.envs:DiscreteVoltVarEnvCap',
)