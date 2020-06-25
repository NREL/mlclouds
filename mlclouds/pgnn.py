# -*- coding: utf-8 -*-
"""
Physics Guided Neural Network
"""
import tensorflow as tf
from tensorflow.keras import layers, optimizers, initializers


class PhysicsGuidedNeuralNetwork:
    """Simple Deep Neural Network with custom physical loss function."""

    def __init__(self, input_shape=1, initializer=None, optimizer=None,
                 learning_rate=0.01):
        """
        Parameters
        ----------
        input_dims : int
            Number of input features.
        output_dims : int
            Number of output labels.
        initializer : tensorflow.keras.initializers
            Instantiated initializer object. None defaults to GlorotUniform
        optimizer : tensorflow.keras.optimizers
            Instantiated neural network optimization object.
            None defaults to Adam.
        learning_rate : float
            Optimizer learning rate.
        """

        if initializer is None:
            initializer = initializers.GlorotUniform()

        self._layers = []
        self._layers.append(layers.Dense(64, kernel_initializer=initializer,
                                         activation=tf.nn.relu,
                                         input_shape=[input_shape]))
        self._layers.append(layers.Dense(64, kernel_initializer=initializer,
                                         activation=tf.nn.relu))
        self._layers.append(layers.Dense(1, kernel_initializer=initializer))

        if optimizer is None:
            self._optimizer = optimizers.Adam(learning_rate=learning_rate)
        else:
            self._optimizer = optimizer

    @property
    def _weights(self):
        weights = []
        for layer in self._layers:
            weights.append(layer.variables[0])
            weights.append(layer.variables[1])
        return weights

    def _get_loss(self, x, y_true):
        out = self.predict(x)
        loss = tf.math.square(out - y_true)
        return loss

    def _get_grad(self, x, y):
        with tf.GradientTape() as tape:
            for layer in self._layers:
                tape.watch(layer.variables)
            loss = self._get_loss(x, y)
            g = tape.gradient(loss, self._weights)
        return g

    def _run_sgd(self, x, y):
        """Run the stochastic gradient descent for one batch of (x, y)"""
        grad = self._get_grad(x, y)
        self._optimizer.apply_gradients(zip(grad, self._weights))

    def fit(self, x, y):
        """Fit the neural network weights to a training batch of x and y"""
        self._run_sgd(x, y)

    def predict(self, features):
        """Run a prediction on input features."""
        out = self._layers[0](features)
        for layer in self._layers[1:]:
            out = layer(out)
        return out
