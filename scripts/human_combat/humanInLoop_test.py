import logging
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))
from runner.human_in_loop import HumanInLoop

def main(args):
    # 创建并运行 HumanInLoop 实例
    loop = HumanInLoop(args)
    loop.run()
    
    
if __name__ == "__main__":
    #logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    
    logging.basicConfig(
        level=logging.DEBUG,               # 设置日志级别为 DEBUG，意味着记录所有级别的日志
        format='%(asctime)s - %(levelname)s - %(message)s',  # 设置日志格式
        filename='debug.log',              # 指定日志文件名
        filemode='w'                        # 'w'表示写入模式，'a'表示追加模式
    )
    
    main(sys.argv[1:])

    
