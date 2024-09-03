'''
Author: BugCrown
Date: 2024-02-23 16:08:30
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

if len(sys.argv) < 2:
    print("Usage: python split_model_generate.py model_file")
    sys.exit(1)

model_file = sys.argv[1]

model, model_input_shape, model_output_shape = LoadModel(model_file, '/CPU:0') 
bottlenecks = FindBottlenecks(model)

save_folder = 'split_model'
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
else:
    for filename in os.listdir(save_folder):
        file_path = os.path.join(save_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            
for index, i in enumerate(bottlenecks):
    save_filename = 'split_at_%s' %i['layer_index']

    head = Model(inputs=model.input, outputs=model.layers[i['layer_index']].output)
    tail = Model(inputs=model.layers[i['layer_index']].output, outputs=model.output)
    
    head.save(os.path.join(save_folder, save_filename + '_head.h5'))
    tail.save(os.path.join(save_folder, save_filename + '_tail.h5'))

save_filename = "configuration"
with open(os.path.join(save_folder, save_filename + '.json'), 'w') as f:
    json.dump(bottlenecks, f)