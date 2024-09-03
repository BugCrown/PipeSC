'''
Author: BugCrown
Date: 2024-01-05 17:53:24
LastEditors: BugCrown
'''
import socket
import struct
import numpy as np
import time
import json
import sys
import os

sys.path.append("../../")
from pipesc.core.transmission import *
from pipesc.core.inference import *
from pipesc.core.splitsc import *

def TCPClient(data, adress):
    sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sc.connect(adress)
    try:
        sc.sendall(data)
        # print("Transmission Success")
    except Exception as e:
        print("Transmission abnomality", e)
    sc.close()
    
if __name__ == '__main__':
    with open(os.path.join('../configuration.json'), 'r') as f:
        tcp_config = json.load(f)

    with open(os.path.join('../split_point_eval/split_model/configuration.json'), 'r') as f:
        split_config = json.load(f)

    ADDR = (tcp_config["HOST"], tcp_config["PORT"])
    BUFFSIZE = tcp_config["BUFFERSIZE"]
    TRANSMAXSIZE = tcp_config["TRANSMAXSIZE"]

    # Test data generation
    max_size = tcp_config["TRANSMAXSIZE"]
    batch_size = []
    for i in range(1, max_size + 1):
        if max_size % i == 0:
            batch_size.append(i)
    for config in split_config:
        shape = config["output_shape"][1:]
        for batch in batch_size:
            whole_shape = [batch] + shape
            test_data = np.ones(whole_shape)
            for _ in range(10):
                # Start sign
                sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sc.connect(ADDR)
                sc.sendall(b"start")
                sc.close()
                
                flat_data = test_data.flatten()
                flat_data = np.concatenate([np.array(whole_shape), flat_data])
                send_data = struct.pack('f' * len(flat_data), *flat_data)
            
                TCPClient(send_data, ADDR)
                time.sleep(5)
        
    # Stop sign
    sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sc.connect(ADDR)
    sc.sendall(b"end")
    sc.close()
