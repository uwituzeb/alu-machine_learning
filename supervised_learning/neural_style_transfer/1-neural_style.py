#!/usr/bin/env python3
"""Create a class NST that performs tasks for Neural Style Transfer"""

import numpy as np
import tensorflow as tf


class NST:
    """Creating a class NST that performs tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """style_image - the image used as a style reference,
        stored as a numpy.ndarray content_image - the image
        used as a content reference, stored as a numpy.ndarray
        alpha - the weight for content cost beta - the weight for style cost"""
        if not isinstance(style_image, np.ndarray) or \
           len(style_image.shape) != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or \
           len(content_image.shape) != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        style_height, style_width, style_channel = style_image.shape
        content_height, content_width, content_channel = content_image.shape

        if style_height <= 0 or style_width <= 0 or style_channel != 3:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        if content_height <= 0 or content_width <= 0 or content_channel != 3:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")

        if (type(alpha) is not float and type(alpha) is not int) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if (type(beta) is not float and type(beta) is not int) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Sets Tensorflow to execute eagerly
        tf.enable_eager_execution()

        # Sets the instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Method that that rescales an image such that its pixels values are
        between 0 and 1 and its largest side is 512 pixels"""

        if not isinstance(image, np.ndarray) or len(image.shape) != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")
        image_height, image_width, image_channel = image.shape
        if image_height <= 0 or image_width <= 0 or image_channel != 3:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        if image_height > image_width:
            h_new = 512
            w_new = int(image_width * (512 / image_height))
        else:
            w_new = 512
            h_new = int(image_height * (512 / image_width))

        resized = tf.image.resize_bicubic(np.expand_dims(image, axis=0),
                                          size=(h_new, w_new))
        rescaled = resized / 255
        rescaled = tf.clip_by_value(rescaled, 0, 1)
        return (rescaled)

    def load_model(self):
        """creates the model used to calculate cost
        the model should use the VGG19 Keras model as a base
        the model input should be the same as the VGG19 input
        the model output should be a list containing the outputs
        of the VGG19 layers listed in style_layers followed by content _layer
        saves the model in the instance attribute model"""
        VGG19_model = tf.keras.applications.VGG19(include_top=False,
                                                  weights='imagenet')
        VGG19_model.save("VGG19_base_model")
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}

        vgg = tf.keras.models.load_model("VGG19_base_model",
                                         custom_objects=custom_objects)

        style_outputs = []
        content_output = None

        for layer in vgg.layers:
            if layer.name in self.style_layers:
                style_outputs.append(layer.output)
            if layer.name in self.content_layer:
                content_output = layer.output

            layer.trainable = False

        outputs = style_outputs + [content_output]

        model = tf.keras.models.Model(vgg.input, outputs)
        self.model = model
