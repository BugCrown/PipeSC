'''
Author: BugCrown
Date: 2024-02-25 16:41:49
LastEditors: BugCrown
'''
import socket
import time
import struct
import numpy as np
import json
import sys
import os

sys.path.append("../../")
from pipesc.core.transmission import *


time_start = time.perf_counter()
time_end = time.perf_counter()

# Receive middle data and infer
def TCPServer(adress, buffer_size, middle_shape_len):
    global time_start
    global time_end
    
    trans = []
    sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sc.bind(adress)
    sc.listen(1)
    while True:
        try: 
            # Receive data from client and get transmission time
            sock, addr=sc.accept()
            tensor_str = b""
            try:
                while True:
                    recv_data = sock.recv(buffer_size)
                    if recv_data:
                        tensor_str += recv_data
                    else:
                        # print('Transmission Complete')
                        break             
            except Exception as e:
                print("Download Abnormality", e)       

            # Start sign
            if tensor_str == b"start":
                time_start = time.perf_counter()
                time_end = time.perf_counter()
                continue
            if tensor_str == b"end":
                break
            trans_data = np.array(struct.unpack('f' * (len(tensor_str) // 4), tensor_str))
            
            batch_size = int(trans_data[0])
            middle_shape = trans_data[0:middle_shape_len]
            middle_shape = [int(x) for x in middle_shape]
            tensor_data = trans_data[middle_shape_len:]
            reshaped_array = tensor_data.reshape(middle_shape)
                     
            time_end = time.perf_counter()
            t = (time_end - time_start) * 1000
            trans.append(
                {
                    'data_shape': middle_shape[1:],
                    'batch_size': batch_size,
                    'transmission_time': t,
                }
            )
            print(t)
            print("next")
        except ConnectionResetError:
            print("Client %s:%s disconnect abnormaly" %addr)
            continue
        except KeyboardInterrupt:
            print("Server closed")
            break
    sc.close()  
    TransTimeEval(trans, "trans")

if __name__ == '__main__':
    with open(os.path.join('../configuration.json'), 'r') as f:
        tcp_config = json.load(f)

    with open(os.path.join('../split_point_eval/split_model/configuration.json'), 'r') as f:
        split_config = json.load(f)

    ADDR = (tcp_config["HOST"], tcp_config["PORT"])
    BUFFSIZE = tcp_config["BUFFERSIZE"]
    TRANSMAXSIZE = tcp_config["TRANSMAXSIZE"]

    tail_input_shape_len = len(split_config[0]["output_shape"])

    TCPServer(ADDR, BUFFSIZE, tail_input_shape_len)           
    
