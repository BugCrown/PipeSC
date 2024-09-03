'''
Author: BugCrown
Date: 2024-03-26 15:52:12
LastEditors: BugCrown
'''
import sys
sys.path.append("../../../")
from pipesc.core.transmission import *
from pipesc.core.inference import *
from pipesc.core.splitsc import *

import socket
import struct
import numpy as np
import time
import json
import sys
import os
import threading
import queue

# Load configuration
with open(os.path.join('../../pipeline_design/pip.json'), 'r') as f:
    pip_config = json.load(f)
with open(os.path.join('../../configuration.json'), 'r') as f:
    tcp_config = json.load(f)

ADDR = (tcp_config["HOST"], tcp_config["PORT"])
BUFFSIZE = tcp_config["BUFFERSIZE"]
SPLITPOINT = pip_config['layer_index']
HEADSIZE = pip_config['head_batch_size']
TRANSSIZE = pip_config['trans_tail_batch_size']

# FIFO (this can be ignored if you have a hardware fifo)
fifo_cache = queue.Queue(HEADSIZE)

def InferAndCache(event, queue, model, input_data, input_shape, batch_size):
    length = input_shape[0]
    i = 0
    while True:
        if i >= length:
            event.set()
            break
        split_input_data = input_data[i:i + batch_size]
        out, head_time = Inference(model, split_input_data, '/CPU:0', "work")
        queue.put(out)
        i += batch_size
        
def HeadClient(event, queue, addr, batch_size):
    while True:
        if event.isSet():
            event.clear()
            sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sc.connect(addr)
            try:
                sc.sendall(b"end")
            except Exception as e:
                print("Transmission abnomality", e)
            sc.close()
            break
        
        # Pack data
        data = queue.get()
        data = data.tolist()
        
        data_len = len(data)
        i = 0
        while True:
            if i >= data_len:
                break
            data_temp = data[i:i+batch_size]
            flat_data = np.array(data_temp).flatten()
            send_data = struct.pack('f' * len(flat_data), *flat_data)
            
            sc = socket(socket.AF_INET, socket.SOCK_STREAM)
            sc.connect(addr)
            try:
                sc.sendall(send_data)
                # print("Transmission Success")
            except Exception as e:
                print("Transmission abnomality", e)
            sc.close()
            i += batch_size

if __name__ == '__main__':
    address = ADDR
    buffer_size = BUFFSIZE
    spilit_point = SPLITPOINT
    head_batch_size = HEADSIZE
    trans_batch_size = TRANSSIZE
    model_path = "../../split_point_eval/split_model/split_at_" + str(spilit_point) + "_head.h5"
    head, head_input_shape, head_output_shape = LoadModel(model_path, "/CPU:0")     
    test_data, test_shape = LoadDataSet("test_dataset.npy")
    
    # Warm up
    for _ in range(5):
        out, head_time = Inference(head, test_data, '/CPU:0', "work")
        
    # Start sign
    sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sc.connect(address)
    sc.sendall(b"start")
    sc.close()
    
    # Inference thread
    inference_event = threading.Event()
    inference_thread = threading.Thread(target=InferAndCache, args=(inference_event, fifo_cache, head, test_data, test_shape, head_batch_size))
    inference_thread.start()
    
    HeadClient(inference_event, fifo_cache, address, head_batch_size)    