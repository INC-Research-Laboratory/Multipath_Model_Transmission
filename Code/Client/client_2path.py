import socket
import pickle
import numpy as np
import threading
import time
import queue
import os
import sys
import torch
import torch.nn as nn
import torchvision.models as models
import struct
import copy

ok_counts = []
lock = threading.Lock()

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

            while True:
                if len(ok_counts) == 2:
                    ok_counts.clear()
                    break
                elif len(ok_counts) == 0:
                    break

        print(f"Thread {thread_id} exit")

def transMission_thread2(thread_id, server_ip, server_port, client_ip, model, num_transmissions, time_list):
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
    server_ports = [PORT_NUM1, PORT_NUM1]
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

    div_model = len(weight)//5
    model_list = []

    start_time = time.time()
    model1 = copy.deepcopy(weight[:div_model])  # 0~80MB (80MB)
    model2 = copy.deepcopy(weight[div_model:])   #80MB ~ 400MB  (320MB)
    end_time = time.time()
    print(f"split model = {end_time-start_time:.3f}")




    print(f"total weight len = {len(weight)}")
    print(f"weight1 len = {len(model1)}")
    print(f"weight2 len = {len(model2)}")

    threads.append(threading.Thread(target=transMission_thread1, args=(0, server_ips[0], server_ports[0], client_ips[0], model1, num_transmissions, time_list[0])))
    threads.append(threading.Thread(target=transMission_thread2, args=(1, server_ips[1], server_ports[1], client_ips[1], model2, num_transmissions, time_list[1])))


    threads[0].start();threads[1].start()
    threads[0].join();threads[1].join()


