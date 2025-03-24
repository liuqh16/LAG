import socket
import atexit
import signal
import sys

class Tacview(object):
    def __init__(self):
        atexit.register(self.cleanup)  # 注册退出清理
        signal.signal(signal.SIGTSTP, self.handle_sigtstp)  # 捕获 Ctrl+Z
        signal.signal(signal.SIGINT, self.handle_sigint)  # 捕获 Ctrl+C
        # 确保程序在：正常退出、Ctrl+C 终止、Ctrl+Z 挂起、异常退出都能正确释放资源
        # Automatically get the local machine's IP address
        self.host = self.get_ip_address() # to show the real ip
        # Default starting port
        self.port = 12345
        self.setup_server()
    
    def handle_sigtstp(self, signum, frame):
        """ 处理 Ctrl+Z 信号 """
        print("\n捕获到 Ctrl+Z，正在清理资源...")
        self.cleanup()
        sys.exit(0)  # 退出程序

    def handle_sigint(self, signum, frame):
        """ 处理 Ctrl+C 信号 """
        print("\n捕获到 Ctrl+C，正在清理资源...")
        self.cleanup()
        sys.exit(0)  # 退出程序

    def get_ip_address(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))  # Google DNS, 只是为了获得正确的IP地址
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    
    def setup_server(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            print(f"Server listening on {self.host}:{self.port}")
            # Output more prominent message
            print("\n" + "*" * 50)
            print("! IMPORTANT: Please open Tacview Advanced, click Record -> Real-time Telemetry, and input the IP address and port !")
            print("*" * 50 + "\n")
            self.connect()
        except Exception as e:
            print(f"Setup error: {e}")
            self.cleanup()
            raise

    def send_data_to_client(self, data):
        try:
            self.client_socket.send(data.encode())
        except Exception as e:
            print(f"Send error: {e}")
            self.reconnect()
            
    def connect(self):
        try:
            print("Waiting for connection...")
            self.client_socket, self.address = self.server_socket.accept()
            print(f"Accepted connection from {self.address}")
            
            # 发送握手数据
            handshake_data = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
            self.client_socket.send(handshake_data.encode())
            
            # 接收客户端响应
            data = self.client_socket.recv(1024)
            print(f"Received data from {self.address}: {data.decode()}")
            
            # 发送头部数据
            header_data = ("FileType=text/acmi/tacview\nFileVersion=2.1\n"
                          "0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n")
            self.client_socket.send(header_data.encode())
            print("Connection established")
            
        except Exception as e:
            print(f"Connection error: {e}")
            self.cleanup()
            raise

    def reconnect(self):
        print("Attempting to reconnect...")
        self.cleanup()
        self.setup_server()

    def cleanup(self):
        try:
            if hasattr(self, 'client_socket') and self.client_socket:
                self.client_socket.close()
                self.client_socket = None
            if hasattr(self, 'server_socket') and self.server_socket:
                self.server_socket.close()
                self.server_socket = None
        except Exception as e:
            print(f"Cleanup error: {e}")

    def __del__(self):
        self.cleanup()