# agent_base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseAgent(ABC):
    def __init__(self, env):
        """
        初始化 BaseAgent 类

        Args:
            env: 环境对象，用于与环境交互，任务通过 env 获取
        """
        self.env = env    # 当前环境对象
        self.task = self.env.task  # 从 env 获取任务对象
        self.action_space = self.set_action_space()  # 设置动作空间
        self.state = None  # 状态初始化
        self.reset()    # 初始化代理状态

    @abstractmethod
    def get_action(self, agent_id):
        """
        获取代理的动作。这个方法应根据代理的策略来生成动作。

        Args:
            agent_id: 当前代理的标识符

        Returns:
            动作：一般是一个 ndarray 或者 list，表示一个或多个动作
        """
        pass

    @abstractmethod
    def reset(self):
        """
        重置代理的状态。可以用来重新初始化代理的状态，或恢复到初始状态。
        """
        pass

    def set_action_space(self):
        """
        获取并设置代理的动作空间。

        Returns:
            动作空间：通常是 gym.spaces 中定义的对象，表示代理可以采取的动作空间
        """
        return self.task.action_space  # 通过任务对象获取动作空间

    def step(self, agent_id):
        """
        执行一步操作，获取代理的动作并在环境中执行。

        Args:
            agent_id: 当前代理的标识符

        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否完成
            info: 其他信息
        """
        action = self.get_action(agent_id)  # 获取代理的动作
        next_state, reward, done, info = self.env.step(action)  # 在环境中执行动作
        self.state = next_state  # 更新代理的状态
        return next_state, reward, done, info
