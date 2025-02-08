import sys
import os
import time
import traceback
import wandb
import socket
import torch
import random
import logging
import numpy as np
from pathlib import Path
import setproctitle
from config import get_config
from runner.share_jsbsim_runner import ShareJSBSimRunner
from envs.JSBSim.envs import SingleCombatEnv, SingleControlEnv, MultipleCombatEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, ShareSubprocVecEnv, ShareDummyVecEnv
from envs.JSBSim.human_agent.HumanAgent import HumanAgent
from envs.JSBSim.tasks.heading_task import HeadingTask  
from scripts.train.train_jsbsim import parse_args, make_train_env, make_eval_env
from runner.tacview import Tacview

class HumanInLoop:
    def __init__(self, args):
        self.args = args
        self.all_args = self.load_args()
        self.tacview = Tacview()
        self.env = self.load_env()
        self.agent = self.load_agent()


    def load_args(self):
        """加载配置文件"""
        parser = get_config()
        all_args = parse_args(self.args, parser)
        return all_args
    
    def load_env(self):
        if self.all_args.env_name == "SingleCombat":
            env = SingleCombatEnv(self.all_args.scenario_name)
        elif self.all_args.env_name == "SingleControl":
            env = SingleControlEnv(self.all_args.scenario_name)
        elif self.all_args.env_name == "MultipleCombat":
            env = MultipleCombatEnv(self.all_args.scenario_name)
        else:
            logging.error("Can not support the " + self.all_args.env_name + "environment.")
            raise NotImplementedError
        # env.seed(self.all_args.seed + rank * 1000)
        return env
    
    def load_agent(self):
        agent = HumanAgent(self.env)  # 不需要显式传递 task，env 中已包含
        return agent



    def run(self):
        """运行主循环"""
        done = False  # 初始化 done 为 False，表示还没有结束
        timestamp = 0 # use for tacview real time render 
        while not done:
            try:
                # 执行一次 step
                observation, reward, done, info = self.agent.step()  # 确保调用 step 方法
                
                # real render with tacview
                render_data = [f"#{timestamp:.2f}\n"]
                for sim in self.env._jsbsims.values():
                    log_msg = sim.log()
                    if log_msg is not None:
                        render_data.append(log_msg + "\n")

                render_data_str = "".join(render_data)
                try:
                    self.tacview.send_data_to_client(render_data_str)
                except Exception as e:
                    logging.error(f"Tacview rendering error: {e}")
                    # 打印调用栈信息
                    logging.error("".join(traceback.format_exc()))

                timestamp += 0.2  # step 0.2s
                # print(timestamp)

                # 可以加入适当的延时控制，避免过快执行
                time.sleep(0.1)  # 设置每一步之间的间隔时间（单位：秒），根据需求调整

            except Exception as e:
                logging.error(f"An error occurred: {e}")
                # 打印完整的调用栈信息
                logging.error("".join(traceback.format_exc()))
                break  # 可选择退出循环



