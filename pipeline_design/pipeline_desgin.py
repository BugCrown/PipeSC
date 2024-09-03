'''
Author: BugCrown
Date: 2024-03-11 14:55:36
LastEditors: BugCrown
'''
import numpy as np
import json
import sys
import os


def pip(head, trans_tail):
    total_batch = 16
    mid_batch = 0
    t_head = 0
    t_total = 0
    # while True:
    #     if total_batch == 0:
    #         break

    #     total_batch -= head[0]
    #     mid_batch += head[0]
    #     t_head += head[1]
    #     if t_total < t_head:
    #         t_total = t_head

    #     while True:
    #         if mid_batch == 0:
    #             break
    #         mid_batch -= trans_tail[0]
    #         t_total += trans_tail[1]
    
    head_batch = head[0]
    trans_tail_batch = trans_tail[0]
    
    head_time = head[1]
    trans_tail_time = trans_tail[1]
    
    if head_time >= head_batch / trans_tail_batch * trans_tail_time:
        t_total = total_batch / head_batch * head_time + head_batch / trans_tail_batch * trans_tail_time
    else:
        t_total = head_time + total_batch / trans_tail_batch * trans_tail_time
    
    return t_total

def pip_min(head, trans_tail):
    t = []
    for i in head:
        for j in trans_tail:
            if j[0] > i[0]:
                break
            t.append([i[0], j[0], pip(i, j)])
    print(t)
    t_min = t[0][2]
    for i in t:
        if i[2] <= t_min:
            pip_min_t = i
            t_min = i[2]
    
    return pip_min_t

with open(os.path.join('../split_point_eval/head.json'), 'r') as f:
    head = json.load(f)
with open(os.path.join('../transmission_eval/trans.json'), 'r') as f:
    trans = json.load(f)
with open(os.path.join('../split_point_eval/tail.json'), 'r') as f:
    tail = json.load(f)
with open(os.path.join('../split_point_eval/split_model/configuration.json'), 'r') as f:
    split_config = json.load(f)
    

pip_t = []

for split_point in split_config:
    head_t = []
    trans_t = []
    tail_t = []
    for head_time in head:
        if head_time['layer_index'] == split_point['layer_index']:
            head_t.append([head_time['batch_size'], head_time['inference_time']])
    for trans_time in trans:
        if trans_time['data_shape'] == split_point['output_shape'][1:]:
            trans_t.append([trans_time['batch_size'], trans_time['trans_time']])
    for tail_time in tail:
        if tail_time['layer_index'] == split_point['layer_index']:
            tail_t.append([tail_time['batch_size'], tail_time['inference_time']]) 

    # print(head_t)
    # print(trans_t)
    # print(tail_t)
    
    trans_tail_t = []
    for index, i in enumerate(trans_t):
        trans_tail_t.append([trans_t[index][0], trans_t[index][1] + tail_t[index][1]])
    
    best_pip = pip_min(head_t, trans_tail_t)
    pip_t.append(
        {
            'layer_index': split_point['layer_index'],
            'head_batch_size': best_pip[0],
            'trans_tail_batch_size': best_pip[1],
            'pip_time': best_pip[2]
        }
        )

pip_t_min = pip_t[0]['pip_time']
for i in pip_t:
    if i['pip_time'] <= pip_t_min:
        pip_min_t = i
        pip_t_min = i['pip_time']
        
print(pip_min_t)
with open('pip.json', 'w') as f:
    json.dump(pip_min_t, f)
    
        
    