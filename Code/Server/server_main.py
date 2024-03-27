### 실험 진행 시 서버 구동 (우선 실행)
import socket
import threading
import pickle
import struct
from collections import OrderedDict
import time
import numpy as np

weight = []
model_mergingTime = threading.Event()
start_time = 0;end_time = 0
mergingTime = []

def server_thread1(server_ip, server_port, server_index, transmission_num):
    chunk_len = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((server_ip, server_port))
        server_socket.listen()

        print(f"Server {server_index} listening on {server_ip}:{server_port}...")
        while True:
            client_socket, _ = server_socket.accept()
            print(f"Client connected {client_socket.getpeername()}")
            try:
                chunk_len = int(client_socket.recv(1024).decode())
                print(f"{client_socket} : {chunk_len}")
                client_socket.send("ok".encode())
                
                for i in range(transmission_num):
                    data_size = struct.unpack('>I', client_socket.recv(4))[0]
                    received_payload = b""
                    remaining_payload_size = data_size
                    #chunk_total = bytearray()
                    while remaining_payload_size != 0:
                        received_payload += client_socket.recv(remaining_payload_size)
                        remaining_payload_size = data_size - len(received_payload)
                    
                    concatModel(received_payload, server_index, chunk_len)
                    client_socket.send("1".encode())
                    print(f"{client_socket.getpeername()} : 1 send")
                
                client_socket.close()
                print(f"server socket[{server_index}] exit.")
                break

            except Exception as e:
                print(f"Error : {str(e)}")
                

            #client_socket.close()
        print(f"server_thread exit")
        server_socket.close()


def server_thread2(server_ip, server_port, server_index, transmission_num):
    chunk_len = 0
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((server_ip, server_port))
        server_socket.listen()

        print(f"Server {server_index} listening on {server_ip}:{server_port}...")
        while True:
            client_socket, _ = server_socket.accept()
            print(f"Client connected {client_socket.getpeername()}")
            #receive_model_params(client_socket, server_socket, server_index)
            try:
                chunk_len = int(client_socket.recv(1024).decode())
                print(f"{client_socket} : {chunk_len}")
                client_socket.send("ok".encode())
                
                for i in range(transmission_num):
                    data_size = struct.unpack('>I', client_socket.recv(4))[0]
                    received_payload = b""
                    remaining_payload_size = data_size
                    #chunk_total = bytearray()
                    while remaining_payload_size != 0:
                        received_payload += client_socket.recv(remaining_payload_size)
                        remaining_payload_size = data_size - len(received_payload)
                
                    concatModel(received_payload, server_index, chunk_len)
                    client_socket.send("1".encode())
                    print(f"{client_socket.getpeername()} : 1 send")
                    
                client_socket.close()
                print(f"server socket[{server_index}] exit.")
                break

            except Exception as e:
                print(f"Error : {str(e)}")
                

        print(f"server_thread exit")
        server_socket.close()



# model concat
def concatModel(model, index, chunk_len):
    global start_time
    global end_time
    
    if index == 0: 
        if len(weight) == 0:
            start_time = time.time()    
        else:
            end_time = time.time()
            
        weight.insert(index, model)
        
        
        ######### multi path ###########
        if len(weight) == 2:
            combined_bytes = bytearray(weight[0])
            combined_bytes += weight[1]
            model = pickle.loads(combined_bytes)
            checkTime = end_time-start_time
            mergingTime.append(checkTime)
            print(f"Model merging success : {checkTime:.3f} sec")
            
            weight.clear()

        ########### single path  ############
        if len(model) == chunk_len:
            model = pickle.loads(model)
            end_time = time.time()
            checkTime = end_time - start_time
            mergingTime.append(checkTime)
            print(model)
            print(f"Model merging success : {checkTime:.3f} sec")
            
            weight.clear()


        
    elif index == 1:
        if len(weight) == 0:
            start_time = time.time()    
        else:
            end_time = time.time()
        
        weight.insert(index, model)
        ######### multi path #############
        if len(weight) == 2:
            combined_bytes = bytearray(weight[0])
            combined_bytes += weight[1]
            model = pickle.loads(combined_bytes)
            checkTime = end_time-start_time
            mergingTime.append(checkTime)
            print(f"Model merging success : {checkTime:.3f} sec")
            
            weight.clear()
            


    

if __name__ == "__main__":
    server_ips = ["Server_IP1", "Server_IP2"]
    server_ports = [PORT_NUM1, PORT_NUM2]
    thread_state = []
    transmission_num = 10
    

    
    thread_state.append(threading.Thread(target=server_thread1, args=(server_ips[0], server_ports[0], 0, transmission_num)))
    thread_state.append(threading.Thread(target=server_thread2, args=(server_ips[1], server_ports[1], 1, transmission_num)))


    thread_state[0].start();thread_state[1].start()
    thread_state[0].join();thread_state[1].join() 

    print(mergingTime)
    print(f"mean = {np.mean(mergingTime)}")
    
    
