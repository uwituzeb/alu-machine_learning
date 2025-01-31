#!/usr/bin/env python3
"""Module for creating the forward propagation graph"""
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Create the forward propagation gragh for the neural network.

    Args:
        x (tf.Tensors): The placeholder for the input data.
        layer_sizes (list): A list containing the number of nodes in each
         layer of the network.
        activations (list): A list containing the activation functions for
         each layer of the network.

    Returns:
        tf.Tensor: The prediction of the network in tensor form.
    """

    layer = x
    for i in range(len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer
