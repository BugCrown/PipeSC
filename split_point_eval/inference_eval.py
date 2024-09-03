'''
Author: BugCrown
Date: 2024-03-08 15:37:09
LastEditors: BugCrown
'''
import sys
sys.path.append("../../")
from pipesc.core.inference import *
from pipesc.core.splitsc import *
from tensorflow.keras.models import Model
import os
import sys
import json
import time
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python inference_eval.py head/tail")
    sys.exit(1)

device = sys.argv[1]

if device == 'head':
    unit = '/CPU:0'
elif device == 'tail':
    unit = '/GPU:0'

with open(os.path.join('./split_model/configuration.json'), 'r') as f:
    split_config = json.load(f)
with open(os.path.join('../configuration.json'), 'r') as f:
    batch_config = json.load(f)
    
max_size = batch_config["TRANSMAXSIZE"]
batch_size = []
for i in range(1, max_size + 1):
    if max_size % i == 0:
        batch_size.append(i)

inference_times = []
for item in split_config:
    model_file_name = os.path.join('./split_model/split_at_' + str(item['layer_index']) + '_' + device + '.h5')
    model, input_shape, output_shape = LoadModel(model_file_name, unit)
    # Test data generation
    input_shape = input_shape[1:]
    for batch in batch_size:
        input_shape_batch = [batch] + list(input_shape)
        print(input_shape_batch)
        input_data = np.ones(input_shape_batch)
        output, inference_time = Inference(model, input_data, unit, 'test')
        inference_times.append(
            {
                'layer_index': item['layer_index'],
                'device': device,
                'batch_size': batch,
                'inference_time': inference_time
            }
        )
        time.sleep(1)

with open(device + '.json', 'w') as f:
    json.dump(inference_times, f)
