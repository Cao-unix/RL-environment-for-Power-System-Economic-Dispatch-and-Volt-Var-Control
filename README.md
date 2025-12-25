# pgym
Gym environments for RL in power systems.

## Installation 
`pip3 install -e .`

## 项目介绍
本项目为电力调度自动化大作业，基于pgym改进的的RL环境

## 代码功能介绍
`pgym\envs\powerflow\econ_dispatch.py` 经济调度环境\\
`pgym\envs\powerflow\continuous_econ_dispatch.py`经济调度子环境，分为直接设置动作和增量设置动作\\

`DQN.py` 离散无功控制任务所用模型\\
`PPO.py` 经济调度任务所用模型\\

`basic.py` 必做部分主程序：离散无功控制\\
`optional1.py` 选做1部分主程序：扩大算例\\
`optional2.py` 选做2部分主程序：经济调度\\
`optional3.py` 选做3部分主程序：在离散无功控制中加入电容器\\




