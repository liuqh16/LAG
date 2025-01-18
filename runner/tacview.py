import socket
import time
class Tacview(object):
    def __init__(self):
        # Prompt user to input IP address and port number
        host = input("Please enter the server IP address: ")
        port = int(input("Please enter the port number: "))

        # Create a socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        server_socket.bind((host, port))

        # Start listening
        server_socket.listen(5)
        print(f"Server listening on {host}:{port}")
        print(f"Please open Tacview Advanced, click Record -> Real-time Telemetry, and input the IP address and port")

        # Wait for client connection
        client_socket, address = server_socket.accept()
        print(f"Accepted connection from {address}")

        self.client_socket = client_socket
        self.address = address

        # Construct handshake data
        handshake_data = "XtraLib.Stream.0\nTacview.RealTimeTelemetry.0\nHostUsername\n\x00"
        # Send handshake data
        client_socket.send(handshake_data.encode())


        # Receive data from the client
        data = client_socket.recv(1024)
        print(f"Received data from {address}: {data.decode()}")
        print("Connection established")

        # Send header data to the client

        data_to_send = ("FileType=text/acmi/tacview\nFileVersion=2.1\n"
                        "0,ReferenceTime=2020-04-01T00:00:00Z\n#0.00\n"
                        )
        client_socket.send(data_to_send.encode())

    def send_data_to_client(self, data):

        self.client_socket.send(data.encode())

