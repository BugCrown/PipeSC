'''
Author: BugCrown
Date: 2024-02-28 13:39:35
LastEditors: BugCrown
'''
# To find forward connections in model
def FindForwardConnection(model):
    forward_connection = []
    for index, layer in enumerate(model.layers):
        forward_layers = layer._outbound_nodes
        if forward_layers:
            forward_layers = [n.outbound_layer for n in forward_layers]
            forward_connection.append(
                {
                    "layer_index": index,
                    "layer_name": layer.name,
                    "forward_layer_index": [model.layers.index(forward_layer) for forward_layer in forward_layers],
                    "forward_layer_name": [forward_layer.name for forward_layer in forward_layers],
                }
            )
    
    return forward_connection

# To find backward connections in model
def FindBackwardConnection(model):
    backward_connection = []
    for index, layer in enumerate(model.layers):
        if index == 0:
            continue
        previous_layers = layer._inbound_nodes
        if previous_layers:
            pre_layers = []
            for n in previous_layers:
                if not isinstance(n.inbound_layers, list):
                    pre_layers.append(n.inbound_layers)
                else:
                    pre_layers = n.inbound_layers

            backward_connection.append(
                {
                    "layer_index": index,
                    "layer_name": layer.name,
                    "previous_layer_index": [model.layers.index(previous_layer) for previous_layer in pre_layers],
                    "previous_layer_name": [previous_layer.name for previous_layer in pre_layers],
                }
            )
    
    return backward_connection

# Models only have main branch, no subbranch
def BranchStartEnd(backward_connection, forward_connection):
    start_nodes_index = []
    end_nodes_index = []
    
    start_end = []
    for layer in forward_connection:
        if len(layer['forward_layer_index']) > 1:
            start_nodes_index.append(layer['layer_index'])
    
    for layer in backward_connection:
        if len(layer['previous_layer_index']) > 1:
            end_nodes_index.append(layer['layer_index'])

    for i, index in enumerate(start_nodes_index):
        start_end.append([start_nodes_index[i], end_nodes_index[i]])
    
    return start_end

# Determine if it is a branch
def isBranch(index, start_end):
    for i in start_end:
        if index >= i[0] and index < i[1]:
            return True 
    return False

# Find natural bottlenecks of the model
def FindBottlenecks(model):
    input_shape = model.input.shape
    input_size = int(1)
    for i in range(len(input_shape) - 1):
        input_size *= input_shape[i + 1]
    
    next_connection = FindForwardConnection(model)
    previous_connection = FindBackwardConnection(model)
    
    start_end = BranchStartEnd(previous_connection, next_connection)
    best_compression = 1
    
    bottlenecks = []
    for index, layer in enumerate(model.layers):
        # Ignore some layers
        if len(layer.output_shape) != len(input_shape):
            continue
        # Treat blocks as a layer
        if isBranch(index, start_end):
            continue
        output_shape = layer.output_shape
        output_size = int(1)
        for i in range(len(output_shape) - 1):
            output_size *= output_shape[i + 1]
        compression = output_size / input_size
        if compression < best_compression:
            best_compression = compression
            bottlenecks.append(
                {
                    'layer_index': model.layers.index(layer),
                    'layer_name': layer.name,
                    'output_size': output_size,
                    'output_shape': output_shape,
                    'compression': compression,
                }
            )
    return bottlenecks