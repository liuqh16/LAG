# 人机交互控制智能体

在该项目中，人类通过键盘控制 HumanAgent，并与环境交互。通过这些交互，智能体执行相应的操作并生成数据，这些数据通过实时渲染传递给 Tacview Advanced，实现对智能体行为和训练过程的可视化。


## 控制智能体方法

    横滚角 (Aileron)：控制飞机的横向运动。
    俯仰角 (Elevator)：控制飞机的纵向运动。
    偏航角 (Rudder)：控制飞机的转向。
    油门 (Throttle)：控制飞机的速度。

键盘操作

    方向键：
        左箭头 (←): 减少横滚角 (Aileron)
        右箭头 (→): 增加横滚角 (Aileron)
        上箭头 (↑): 增加俯仰角 (Elevator)
        下箭头 (↓): 减少俯仰角 (Elevator)

    Z 和 X 键：
        Z: 减少偏航角 (Rudder)
        X: 增加偏航角 (Rudder)

    Page Up 和 Page Down 键：
        Page Up: 增加油门 (Throttle)
        Page Down: 减少油门 (Throttle)

## HumanInLoop类

达到类似控制类的效果，将输入参数、仿真环境、人类智能体、任务等集成到人在回路类中，方便后续集成和开发


## 任务介绍

### HumanFreeFlyTask

顾名思义，旨在让玩家控制飞机自由的飞行，启动脚本：进入 scripts/ 目录并运行以下命令启动人机交互控制：

```shell
cd scripts/
bash human_free_fly.sh
```

### HumanSingleCombatTask(TODO)



## 注意事项

Tacview 要求：此功能需要购买并安装 Tacview Advanced。

## TODO

- 优化键盘控制体验：

        改进控制响应速度，使其更加平滑。
        调整各个控制维度的灵敏度，使操作更加自然。

- 设计更多的任务 (Task)：

        除了现有的控制任务，可以考虑加入其他类型的任务，如目标识别、躲避敌机等，以增强智能体训练的多样性。

- 实现与自己训练的智能体对战：

        开发对战模式，允许人类与自己训练的智能体进行交互，进行对抗性训练。