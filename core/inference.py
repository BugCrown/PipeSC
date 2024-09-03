'''
Author: BugCrown
Date: 2024-02-28 13:39:35
LastEditors: BugCrown
'''
from . import _tf, _np, _t

# Load model and output some information about the model
def LoadModel(file_name, unit):
    with _tf.device(unit):
        model = _tf.keras.models.load_model(file_name)
    input_shape = model.input_shape
    output_shape = model.output_shape
    
    return model, input_shape, output_shape

# Load test dataset
def LoadDataSet(file_name):
    test_data = _np.load(file_name)
    test_shape = _np.shape(test_data)
    return test_data, test_shape

# Infer the model and time on device
def Inference(model, input, unit, sign):
    if sign == "test": 
        # Warm up
        with _tf.device(unit):
            for _ in range(10):
                output = model.predict(input)
                
            # Reat 10 times
            start_time = _t.perf_counter()
            for _ in range(100):
                output = model.predict(input)
            end_time = _t.perf_counter()
            inference_time = (end_time - start_time) * 10 # milliseconds
    elif sign == "work":
        start_time = _t.perf_counter()
        with _tf.device(unit):
            output = model.predict(input) 
        end_time = _t.perf_counter()
        inference_time = (end_time - start_time) * 1000 # milliseconds
    return output, round(inference_time, 2)