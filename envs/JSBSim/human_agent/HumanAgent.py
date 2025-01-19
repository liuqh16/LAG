import atexit
import numpy as np
import sys
from ..envs.singlecontrol_env import SingleControlEnv
from gymnasium import spaces
from .agent_base import BaseAgent
import curses
import threading

class HumanAgent(BaseAgent):
    def __init__(self, env: SingleControlEnv):
        super().__init__(env)  # 直接传入 env，并从 env 获取 task
        self.env = env
        self.action_space = self.task.action_space  # 获取动作空间
        self.action_dim = len(self.action_space.nvec)  # 动作维度（4个控制命令）
        self.aileron = 20   # 控制横滚角 (Aileron) [0, 40] -> [-1., 1.] 
        self.elevator = 20  # 控制俯仰角 (Elevator) [0, 40] -> [-1., 1.] 
        self.rudder = 20    # 控制偏航角 (Rudder) [0, 40] -> [-1., 1.] 
        self.throttle = 15  # 控制油门 (Throttle)  [0, 30] -> [0.4, 0.9] 

        self.input_thread = None
        self.stop_event = threading.Event()

        # 注册清理操作
        atexit.register(self.cleanup)

        self.start_input_thread()

    def cleanup(self):
        """确保线程停止"""
        print("Cleaning up HumanAgent...")
        self.stop_input_thread()
        
    def start_input_thread(self):
        """启动输入线程"""
        if self.input_thread is None:
            #print("Initializing input thread...")
            self.input_thread = threading.Thread(target=self.keyboard_input)
            self.input_thread.daemon = True  # 设置为守护线程，程序退出时自动退出
            self.input_thread.start()
            # print("Input thread started.")

    def stop_input_thread(self):
        """停止输入线程"""
        if self.input_thread is not None:
            if self.input_thread.is_alive():
                #print("Stopping input thread...")
                self.stop_event.set()  # 设置停止事件，通知线程退出
                self.input_thread.join()  # 等待线程退出
                #print("Input thread has been stopped.")
            else:
                print("Input thread is not alive. Nothing to stop.")
        else:
            print("Input thread is None. Can't stop it.")
            
    def keyboard_input(self):
        """
        通过键盘输入来调整控制参数的线程函数
        """
        stdscr = curses.initscr()
        curses.cbreak()
        stdscr.keypad(1)

        try:
            # Create a window to display control parameters separately
            height, width = stdscr.getmaxyx()
            control_win = curses.newwin(5, width, 0, 0)  # Control panel at the top
            info_win = curses.newwin(height-5, width, 5, 0)  # Info panel below control panel

            while not self.stop_event.is_set():  # 使用事件控制线程停止
                control_win.clear()
                control_win.addstr(f"Aileron: {self.aileron}  Elevator: {self.elevator}  Rudder: {self.rudder}  Throttle: {self.throttle}\n")
                control_win.addstr("Use Arrow keys to control Aileron/Elevator, Z/X for Rudder, + for Throttle Up, - for Throttle Down.\n")

                # Clear info window and add other logs there
                info_win.clear()
                info_win.addstr("current_step: 151 target_heading: 162.16831804861158\n")
                info_win.addstr("target_altitude_ft: 15000 target_velocities_u_mps: 334.23289082450526\n")
                info_win.addstr("AircraftSimulator:A0100 is deleted!\n")  # Example log info

                key = stdscr.getch()
                # 左右控制横滚角
                if key == curses.KEY_LEFT and self.aileron < 40:
                    self.aileron -= 1
                elif key == curses.KEY_RIGHT and self.aileron > 0:
                    self.aileron += 1
                # 上下控制俯仰角
                elif key == curses.KEY_UP and self.elevator > 0:
                    self.elevator -= 1
                elif key == curses.KEY_DOWN and self.elevator < 40:
                    self.elevator += 1

                elif key == ord('z') and self.rudder > 0:
                    self.rudder -= 1
                elif key == ord('x') and self.rudder < 40:
                    self.rudder += 1

                elif key == ord('+') and self.throttle < 29:
                    self.throttle += 1
                elif key == ord('-') and self.throttle > 0:
                    self.throttle -= 1

                control_win.refresh()  # Refresh control window
                info_win.refresh()  # Refresh info window

        finally:
            curses.endwin()

    def get_action(self):
        # 返回动作数组
        action = np.array([self.aileron, self.elevator, self.rudder, self.throttle])
        return action.reshape(1, -1)  # 转换为二维数组
    
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
        # print("Resetting HumanAgent...")
        self.env.reset()  # 调用环境的 reset 方法
        return self.env.get_obs()  # 返回环境的初始观察状态
    
    def __del__(self):
        """析构函数，确保线程停止"""
        # print("Cleaning up HumanAgent...")
        self.stop_input_thread()  # 显式调用 stop_input_thread
