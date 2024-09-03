'''
Author: BugCrown
Date: 2024-02-28 13:39:35
LastEditors: BugCrown
'''
from . import _np, _js

def TransTimeEval(trans_list, save_filename):
    grouped_data = {}
    for item in trans_list:
        key = (tuple(item['data_shape']), item['batch_size'])
        if key in grouped_data:
            grouped_data[key].append(item['transmission_time'])
        else:
            grouped_data[key] = [item['transmission_time']]

    # Calculate the average transfer time per batch size
    average_times = []
    for key, times in grouped_data.items():
        average_time = sum(times) / len(times)
        average_times.append(
            {
                'data_shape': key[0],
                'batch_size': key[1],
                'trans_time': average_time              
            }
        )

    with open(save_filename + '.json', 'w') as f:
        _js.dump(average_times, f)

    print("Normalized transmission times:", average_times)
    
# Reduce the flat array
def ReduceFlatArray(array, shape):
    # Get batch size
    array_len = len(array)
    batch_size = array_len // _np.prod(shape[1:])
    reshaped_array = array.reshape((batch_size,)+shape[1:])
    
    return reshaped_array