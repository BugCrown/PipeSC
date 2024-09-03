'''
Author: BugCrown
Date: 2024-01-05 15:22:57
LastEditors: BugCrown
'''
import socket
import json
import time
import tensorflow as tf
import sys
import pickle
import struct
import zlib
import numpy as np

from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
from utils import *

HOST = '192.168.1.68'
PORT = 9001
ADDR = (HOST, PORT)
BUFFSIZE = 1024

# Receive middle data and infer
def TCPServer(adress, buffer_size, model, middle_shape):
    start_time = 0
    sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sc.bind(adress)
    sc.listen(1)
    while True:
        try: 
            # Receive data from client and get transmission time
            sock, addr=sc.accept()
            # print("Connection from: %s:%s" %addr)
            
            end_time_1 = time.perf_counter()
            t = end_time_1 - start_time
                    
            # trans_start_time = time.perf_counter() 
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
            # else:
            #     print("Download Success")
            # trans_end_time = time.perf_counter()
            # trans_time = (trans_end_time - trans_start_time) * 100 # milliseconds
            # print("Transmission time: %.2fms" %trans_time)   

            # Inference and time on server
            # print(len(tensor_str))
            # tensor_str = zlib.decompress(tensor_compressed_str)         

            # Start sign
            if tensor_str == b"start":
                start_time = time.perf_counter()
                continue
            tensor_data = struct.unpack('f' * (len(tensor_str) // 4), tensor_str)
            tensor_data = ReduceFlatArray(np.array(tensor_data), middle_shape)
            
            output, tail_time = Inference(model, tensor_data, "/GPU:0", "work")
            print("Inference time: %.2fms" %tail_time)
            preds = decode_predictions(output, top=3)
            
            for i in preds:
                print(i)
            end_time = time.perf_counter()
            t = end_time - start_time
            print("total time: %.2fms" %(t * 100))
                
        except ConnectionResetError:
            print("Client %s:%s disconnect abnormaly" %addr)
            continue
        except KeyboardInterrupt:
            print("Server closed")
            break
    sc.close()
        
        

if __name__ == '__main__':
    tail, tail_input_shape, tail_output_shape = LoadModel("split_at_16_tail.h5", "/GPU:0")
    
    # Warm up
    input_data = tf.ones((16,)+tail_input_shape[1:])
    with tf.device("/GPU:0"):
        for _ in range(10):
            output = tail.predict(input_data)
            
    TCPServer(ADDR, BUFFSIZE, tail, tail_input_shape)