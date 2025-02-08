import atexit
import logging
import time
import numpy as np
from ..envs.singlecontrol_env import SingleControlEnv
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
            
    
    def refresh_windows(self, control_win, info_win):
        """
        刷新控制面板和信息面板窗口
        """
        control_win.refresh()  # 刷新控制面板窗口
        info_win.refresh()  # 刷新信息面板窗口

    def keyboard_input(self):
        stdscr = curses.initscr()  # 初始化 curses 终端窗口 stdscr，用于接收键盘输入并控制终端界面显示。
        curses.cbreak()            # cbreak 模式，使程序可以立即响应键盘输入，而不需要按下 Enter
        stdscr.keypad(1)           # 支持解析功能键（如方向键、Page Up/Down 等）。

        try:
            height, width = stdscr.getmaxyx()
            control_win = curses.newwin(5, width, 0, 0)         # 控制窗口
            info_win = curses.newwin(height-5, width, 5, 0)     # 信息窗口

            while not self.stop_event.is_set():  # 进入无限循环，直到 self.stop_event 事件被触发（可能是外部中断信号）。
                control_win.clear()              # 清空 control_win 窗口，准备更新新的控制信息。

                # 更新控制面板显示
                control_win.addstr(f"Aileron: {self.aileron}  Elevator: {self.elevator}  Rudder: {self.rudder}  Throttle: {self.throttle}\n")
                control_win.addstr("Use Arrow keys to control Aileron/Elevator, Z/X for Rudder, PgUp for Throttle Up, PgDn for Throttle Down.\n")

                info_win.clear()

                # 设置一个适当的超时时间，单位是毫秒，例如 500 毫秒
                stdscr.timeout(500)         # 设置 stdscr 的超时时间为 500 毫秒。如果在 500 毫秒内没有键盘输入，则返回key为 -1，从而避免程序阻塞。


                key = stdscr.getch()        # 读取用户按下的键。

                self.update_action(key)     # 更新动作
                
                # 限制处理速度，避免过快刷新
                time.sleep(0.05)

                # 刷新窗口
                self.refresh_windows(control_win, info_win)

        finally:
            curses.endwin()
    
    def update_action(self, key):
        # 左右控制横滚角
        if key == curses.KEY_LEFT and self.aileron < 40:
            self.aileron -= 1
        elif key == curses.KEY_RIGHT and self.aileron > 0:
            self.aileron += 1
        # 上下控制俯仰角
        elif key == curses.KEY_UP and self.elevator > 0:
            self.elevator += 1
        elif key == curses.KEY_DOWN and self.elevator < 40:
            self.elevator -= 1
        # 控制其他操作
        elif key == ord('z') and self.rudder > 0:
            self.rudder -= 1
        elif key == ord('x') and self.rudder < 40:
            self.rudder += 1
        elif key == curses.KEY_PPAGE and self.throttle < 29:
            self.throttle += 1
        elif key == curses.KEY_NPAGE and self.throttle > 0:
            self.throttle -= 1
        elif key == -1:
            self.elevator = 20
            self.aileron = 20
            self.rudder = 20
            self.throttle = 15
    
    def get_action(self):
        # 返回动作数组
        # 在创建动作数组之前，打印变量的值
        action = np.array([self.aileron, self.elevator, self.rudder, self.throttle])
        return action.reshape(1, -1)  # 转换为二维数组
    
    def step(self):
        """
        Perform an action step in the environment based on the user input.
        """
        action = self.get_action()  # 获取动作
        observation, reward, done, info = self.env.step(action)  # 执行动作
        return observation, reward, done, info
    
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
