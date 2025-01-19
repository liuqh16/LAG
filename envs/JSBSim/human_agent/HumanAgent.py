# HumanAgent.py
import numpy as np
import sys
from ..envs.singlecontrol_env import SingleControlEnv
from gymnasium import spaces
from .agent_base import BaseAgent

class HumanAgent(BaseAgent):
    def __init__(self, env: SingleControlEnv):
        super().__init__(env)  # 直接传入 env，并从 env 获取 task
        self.env = env
        self.action_space = self.task.action_space  # 获取动作空间
        self.action_dim = len(self.action_space.nvec)  # 动作维度（4个控制命令）

    def get_action(self):
        # 假设固定姿态角和油门
        aileron = 20  # 固定 Aileron（方向舵）
        elevator = 15  # 固定 Elevator（升降舵）
        rudder = 30  # 固定 Rudder（方向舵）
        throttle = 15  # 固定油门

        # 将固定的动作组合为一个动作数组
        action = np.array([aileron, elevator, rudder, throttle])

        # 如果动作是1维的数组，转换为2维数组
        if len(action.shape) == 1:  # 如果是1维数组，转换为2维数组
            action = action.reshape(1, -1)  # 转换为二维数组，形状为 (1, action_dim)

        return action



    def step(self):
        """
        Perform an action step in the environment based on the user input.
        """
        action = self.get_action()  # 获取动作
        observation, reward, done, info = self.env.step(action)  # 执行动作
        return observation, reward, done, info


    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value."""
        norm_act = np.zeros(4)

        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
    
        return norm_act
    
    def reset(self):
        """
        Reset the agent. This method is required for the abstract class.
        You can initialize the agent state here if needed.
        """
        print("Resetting HumanAgent...")
        # 如果需要可以初始化 agent 状态，或者你可以直接重置环境
        self.env.reset()  # 调用环境的 reset 方法
        return self.env.get_obs()  # 返回环境的初始观察状态
