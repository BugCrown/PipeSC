'''
Author: BugCrown
Date: 2024-01-05 17:53:24
LastEditors: BugCrown
'''
import socket
import json
import time
import tensorflow as tf
import sys
import pickle
import struct
import threading
import queue
import numpy as np
from utils import *

# FIFP Cache
fifo_cache = queue.Queue(10)

HOST = '192.168.1.68'
PORT = 9001
ADDR = (HOST, PORT)
ENCODING = 'utf-8'
BUFFSIZE = 1024

def InferAndCache(event, queue, model, input_data, input_shape, batch_size):
    length = input_shape[0]
    i = 0
    while True:
        if i >= length:
            event.set()
            break
        # print("infer start" + str(time.perf_counter()))
        split_input_data = input_data[i:i + batch_size]
        out, head_time = Inference(model, split_input_data, '/CPU:0', "work")
        print("Inference time: %.2fms" %head_time)
        # print("%.2f" %head_time)
        out = out.tolist()
        flat_data = np.array(out).flatten()
        send_data = struct.pack('f' * len(flat_data), *flat_data)
        queue.put(send_data)
        i += batch_size
        # print("infer end" + str(time.perf_counter())) 



def TCPClient(event, queue):
    while True:
        if event.isSet():
            event.clear()
            break
        # print("tcp" + str(time.perf_counter()))
        data = queue.get()
        # flat_data = np.array(data).flatten()
        # print("t1 %.2f" %(time.perf_counter()*100))
        # send_data = struct.pack('f' * len(flat_data), *flat_data)
        # print("t2 %.2f" %(time.perf_counter()*100))
        
        sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sc.connect(ADDR)
        try:
            sc.sendall(data)
            # print("Transmission Success")
        except Exception as e:
            print("Transmission abnomality", e)
        sc.close()


if __name__ == '__main__':
    head, head_input_shape, head_output_shape = LoadModel("split_at_16_head.h5", "/CPU:0")  
    test_data, test_shape = LoadDataSet("test_dataset.npy")
    batch_size = 1
    
    # Warm up
    for _ in range(5):
        out, head_time = Inference(head, test_data, '/CPU:0', "work")
    
    # Start sign
    sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sc.connect(ADDR)
    sc.sendall(b"start")
    sc.close()
    
    # Inference thread
    inference_event = threading.Event()
    inference_thread = threading.Thread(target=InferAndCache, args=(inference_event, fifo_cache, head, test_data, test_shape, batch_size))
    inference_thread.start()
    # TCP thread
    # tcp_thread = threading.Thread(target=TCPClient, args=(inference_event, fifo_cache, ))
    # tcp_thread.start()
    TCPClient(inference_event, fifo_cache)
