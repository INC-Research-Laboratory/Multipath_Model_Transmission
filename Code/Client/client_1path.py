### Single Path 실험 시 실행
import socket
import pickle
import numpy as np
import threading
import time
import torch
import torch.nn as nn
import torchvision.models as models

ok_counts = []
lock = threading.Lock()
event = threading.Event()



# model define
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(64 * 56 * 56, 500)
        self.relu3 = nn.ReLU()

        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)

        return x

def transMission_thread1(thread_id, server_ip, server_port, client_ip, model, num_transmissions, time_list):
    global ok_counts
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.bind((client_ip, server_port))
        client_socket.connect((server_ip, server_port))
        client_socket.send(str(len(model)).encode())
        print(client_socket.recv(1024).decode())
        for i in range(num_transmissions):
            start_time = time.time()
            client_socket.send(len(model).to_bytes(4, byteorder='big'))
            client_socket.send(model)
            end_time = time.time()
            this_time = end_time - start_time
            time_list.append(this_time)


            lock.acquire()
            ok_counts.append(client_socket.recv(1024).decode())
            lock.release()
            print(f"{client_socket.getpeername()} : ok")

            print(f"[{client_ip} -> {server_ip} : {i}] transmission time : {this_time:.3f} sec")

        print(f"Thread {thread_id} exit")
        event.set()


def transMission_thread2(thread_id, server_ip, server_port, client_ip, model, num_transmissions, time_list):
    global ok_counts
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.bind((client_ip, server_port))
        client_socket.connect((server_ip, server_port))
        client_socket.send("0".encode())
        print(client_socket.recv(1024).decode())

        if(client_ip == "Client_IP2"):
            print("Wait...")
            while not event.is_set():
                time.sleep(5)
                print(f"{thread_id} wait...")
            print(f"Thread {thread_id} exit")
            exit(1)

        for i in range(num_transmissions):
            start_time = time.time()
            client_socket.send(len(model).to_bytes(4, byteorder='big'))
            client_socket.send(model)
            end_time = time.time()
            this_time = end_time - start_time
            time_list.append(this_time)


            lock.acquire()
            ok_counts.append(client_socket.recv(1024).decode())
            lock.release()
            print(f"{client_socket.getpeername()} : ok")

            print(f"[{client_ip} -> {server_ip} : {i}] transmission time : {this_time:.3f} sec")

            while True:
                if len(ok_counts) == 2:
                    ok_counts.clear()
                    break
                elif len(ok_counts) == 0:
                    break

        print(f"Thread {thread_id} exit")


if __name__ == "__main__":
    server_ips = ["Server_IP1", "Server_IP2"]
    client_ips = ["Client_IP1", "Client_IP2"]  # 0.~ (2.4, n mode)  7.~(5, ac mode)
    server_ports = [PORT_NUM1, PORT_NUM2]
    num_transmissions = 10
    threads = []
    filePath_list = []
    time_list = [[], []]


    model = ConvNet()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConvNet().to(device)
    model = model.state_dict().items()
    dict_weight = dict(model)
    print(dict_weight)
    weight = pickle.dumps(dict_weight)

    print(f"total weight len = {len(weight)}")


    threads.append(threading.Thread(target=transMission_thread1, args=(0, server_ips[0], server_ports[0], client_ips[0], weight, num_transmissions, time_list[0])))
    threads.append(threading.Thread(target=transMission_thread2, args=(1, server_ips[1], server_ports[1], client_ips[1], 0, num_transmissions, time_list[1])))


    threads[0].start();threads[1].start()
    threads[0].join();threads[1].join()

