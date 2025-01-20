import socket
import time

class Tacview(object):
    def __init__(self):
        # Automatically get the local machine's IP address
        host = socket.gethostbyname(socket.gethostname())
        # Default starting port
        port = 12345

        # Create a socket and store it as an instance variable
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Allow reusing the address/port
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.server_socket.bind((host, port))

        # Start listening
        self.server_socket.listen(5)
        print(f"Server listening on {host}:{port}")
        # Output more prominent message
        print("\n" + "*" * 50)
        print("! IMPORTANT: Please open Tacview Advanced, click Record -> Real-time Telemetry, and input the IP address and port !")
        print("*" * 50 + "\n")

        # Wait for client connection
        self.client_socket, self.address = self.server_socket.accept()
        print(f"Accepted connection from {self.address}")

        # Construct handshake data
        handshake_data = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
        # Send handshake data
        self.client_socket.send(handshake_data.encode())

        # Receive data from the client
        data = self.client_socket.recv(1024)
        print(f"Received data from {self.address}: {data.decode()}")
        print("Connection established")

        # Send header data to the client
        data_to_send = ("FileType=text/acmi/tacview\nFileVersion=2.1\n"
                        "0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n"
                        )
        self.client_socket.send(data_to_send.encode())

    def send_data_to_client(self, data):
        self.client_socket.send(data.encode())

    def __del__(self):
        """Destructor: Ensure sockets are closed when the object is deleted."""
        try:
            # Close client socket if it exists
            if hasattr(self, 'client_socket') and self.client_socket:
                self.client_socket.close()    
            # Close server socket if it exists
            if hasattr(self, 'server_socket') and self.server_socket:
                self.server_socket.close()
        except Exception as e:
            print(f"Error during cleanup: {e}")