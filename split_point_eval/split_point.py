'''
Author: BugCrown
Date: 2024-02-28 13:48:36
LastEditors: BugCrown
'''
import sys
sys.path.append("../../")
from pipesc.core.inference import *
from pipesc.core.splitsc import *
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def HeadInference(head, input):
    with tf.device('/CPU:0'):
        out = head(input)
    return out

def TailInference(tail, input):
    with tf.device('/GPU:0'):
        out = tail(input)
    return out

model_file = "../model/mobilenet_v2.h5"
model, model_input_shape, model_output_shape = LoadModel(model_file, '/CPU:0')  
input_shape = (1,) + model_input_shape[1:]
input_data = tf.ones(input_shape)

save_folder = './%s' %model.name.split('.')[0]
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    
bottlenecks = FindBottlenecks(model)

bandwidth = np.arange(1 * 10 ** 6, 512 * 10 ** 6, 1 * 10 ** 6)
PLOT_LINESTYLES = ['-', '--', '-.', ':']
PLOT_COLOR = ['red', 'dodgerblue', 'lawngreen', 'magenta', 'coral']

print(bottlenecks)
# Assuming the head device has enough resource
plt.figure(figsize=(20, 5))
plt.ylim(2,20)
save_filename = 'split_eval_batch_1'
for index, layer in enumerate(bottlenecks):
    linestyle = PLOT_LINESTYLES[index % len(PLOT_LINESTYLES)]  
    color = PLOT_COLOR[index % len(PLOT_COLOR)]
    
    with tf.device('/CPU:0'):
        head = Model(inputs=model.input, outputs=model.layers[layer['layer_index']].output)
    with tf.device('/GPU:0'):
        tail = Model(inputs=model.layers[layer['layer_index']].output, outputs=model.output)
    # Warm up
    for _ in range(10):
        mid = HeadInference(head, input_data)
        output_data = TailInference(tail, mid)
        
    # Consider inference time variaion
    t_10 = []
    for _ in range(10):
        start_time = time.perf_counter()     
        for __ in range(10):
            mid = HeadInference(head, input_data)
            output_data = TailInference(tail, mid)
        end_time = time.perf_counter()
        t = (end_time - start_time) / 10 * 100
        t_10.append(t)
            
    t_max = np.max(t_10)
    t_max += layer['output_size'] * 4 / bandwidth * 100
    t_min = np.min(t_10)
    t_min += layer['output_size'] * 4 / bandwidth * 100
    t_mean = np.mean(t_10)
    t_mean += layer['output_size'] * 4 / bandwidth * 100
    
    plt.plot(
        bandwidth / 10 ** 6,
        t_max,
        label="Split at \nlayer %s" %layer['layer_index'],
        linestyle='-',
        color=color,
    )
    plt.plot(
        bandwidth / 10 ** 6,
        t_min,
        linestyle='-',
        color=color,
    )
    plt.fill_between(
        bandwidth / 10 ** 6,
        t_min,
        t_max,
        facecolor=color,
        alpha=0.5,
    )
plt.xlabel('Data Rate (MBps)')
plt.ylabel('Inference Time (ms)')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
plt.savefig(os.path.join(save_folder, save_filename))
tf.keras.backend.clear_session()