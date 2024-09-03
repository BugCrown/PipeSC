'''
Author: BugCrown
Date: 2024-03-29 17:36:53
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
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions

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
time_start = time.perf_counter()
time_end = time.perf_counter()

def TailServer(event, queue, condition, adress, buffer_size):
    global time_start
    global time_end
    
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
                        break             
            except Exception as e:
                print("Download Abnormality", e)
                
            if tensor_str == b"start":
                time_start = time.perf_counter()
                time_end = time.perf_counter()
                print("a")
                continue           
            if tensor_str == b"end":
                event.set()
                break
              
            with condition:
                queue.put(tensor_str)
                condition.notify()  # Notify inference thread
                
        except ConnectionResetError:
            print("Client %s:%s disconnect abnormaly" %addr)
            continue
        except KeyboardInterrupt:
            print("Server closed")
            break
    sc.close()    
    
def FetchAndInfer(event, queue, condition, model, middle_shape):
    global time_end
    while True:
        with condition:
            while queue.empty():
                condition.wait()
            tensor_str = queue.get()
        
        tensor_data = np.array(struct.unpack('f' * (len(tensor_str) // 4), tensor_str))
        array_len = len(tensor_data)
        batch_size = array_len // np.prod(middle_shape[1:])
        reshaped_array = tensor_data.reshape((batch_size,)+middle_shape[1:])
        output, tail_time = Inference(model, reshaped_array, "/GPU:0", "work")
        preds = decode_predictions(output, top=3)
        time_end = time.perf_counter()
        t = time_end - time_start
        print("time: %.2fms" %(t * 1000)) 
        
        for i in preds:
            print(i)
        
        if event.isSet():
            time_end = time.perf_counter()
            t = time_end - time_start
            print("total time: %.2fms" %(t * 1000)) 
            
            event.clear()
            break
        
if __name__ == '__main__':
    address = ADDR
    buffer_size = BUFFSIZE
    spilit_point = SPLITPOINT
    head_batch_size = HEADSIZE
    trans_batch_size = TRANSSIZE
    model_path = "../../split_point_eval/split_model/split_at_" + str(spilit_point) + "_tail.h5"    
    tail, tail_input_shape, tail_output_shape = LoadModel(model_path, "/GPU:0")
    
    # Warm up
    input_data = tf.ones((16,) + tail_input_shape[1:])
    for _ in range(10):
        output, tail_time = Inference(tail, input_data, "/GPU:0", "work")
        
    fifo_condition = threading.Condition()
    inference_event = threading.Event()
    inference_thread = threading.Thread(target=FetchAndInfer, args=(inference_event, fifo_cache, fifo_condition, tail, tail_input_shape))
    inference_thread.start()        
    TailServer(inference_event, fifo_cache, fifo_condition, address, buffer_size)
               