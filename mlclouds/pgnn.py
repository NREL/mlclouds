# -*- coding: utf-8 -*-
"""
Physics Guided Neural Network
"""
import time
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras import layers, optimizers, initializers


logger = logging.getLogger(__name__)


class PhysicsGuidedNeuralNetwork:
    """Simple Deep Neural Network with custom physical loss function."""

    def __init__(self, p_fun, loss_weights=(0.5, 0.5),
                 input_dims=1, output_dims=1,
                 initializer=None, optimizer=None,
                 learning_rate=0.01):
        """
        Parameters
        ----------
        p_fun : function
            Physics function to guide the neural network loss function.
            This function must take (y_predicted, y_true, p, **p_kwargs)
            as arguments with datatypes (tf.Tensor, np.ndarray, np.ndarray).
            The function must return a tf.Tensor object with a single numeric
            loss value (output.ndim == 0).
        loss_weights : tuple
            Loss weights for the neural network y_predicted vs. y_true
            and for the p_fun loss, respectively. For example,
            loss_weights=(0.0, 1.0) would simplify the PGNN loss function
            to just the p_fun output.
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

        self._p_fun = p_fun
        self._loss_weights = loss_weights
        self._layers = []
        self._optimizer = None
        self._history = None

        if initializer is None:
            initializer = initializers.GlorotUniform()

        self._layers.append(layers.InputLayer(input_shape=[input_dims]))
        self._layers.append(layers.Dense(64, kernel_initializer=initializer,
                                         activation=tf.nn.relu))
        self._layers.append(layers.Dense(64, kernel_initializer=initializer,
                                         activation=tf.nn.relu))
        self._layers.append(layers.Dense(output_dims,
                                         kernel_initializer=initializer))

        if optimizer is None:
            self._optimizer = optimizers.Adam(learning_rate=learning_rate)
        else:
            self._optimizer = optimizer

    @staticmethod
    def _check_shapes(x, y):
        """Check the shape of two input arrays for usage in this NN."""
        assert len(x.shape) == 2
        assert len(y.shape) == 2
        assert len(x) == len(y)
        return True

    @staticmethod
    def seed(s=0):
        """Set the random seed for reproducable results."""
        np.random.seed(s)
        tf.random.set_seed(s)

    @property
    def history(self):
        """Get the training history dataframe (None if not yet trained)."""
        return self._history

    @property
    def layers(self):
        """Get a list of the NN layers."""
        return self._layers

    @property
    def weights(self):
        """Get a list of layer weights for gradient calculations."""
        weights = []
        for layer in self._layers:
            weights += layer.variables
        return weights

    def loss(self, y_predicted, y_true, p, p_kwargs):
        """Calculate the loss function by comparing model-predicted y to y_true

        Parameters
        ----------
        y_predicted : tf.Tensor
            Model-predicted output data in a 2D tensor.
        y_true : np.ndarray
            Known output data in a 2D array.
        p : np.ndarray
            Supplemental feature data for the physics loss function in 2D array
        p_kwargs : None | dict
            Optional kwargs for the physical loss function self._p_fun.

        Returns
        -------
        loss : tf.tensor
            Sum of the NN loss function comparing the y_predicted against
            y_true and the physical loss function (self._p_fun) with
            respective weights applied.
        """

        if p_kwargs is None:
            p_kwargs = {}

        nn_loss = tf.math.reduce_mean(tf.math.abs(y_predicted - y_true))

        p_loss = self._p_fun(y_predicted, y_true, p, **p_kwargs)

        loss = (self._loss_weights[0] * nn_loss
                + self._loss_weights[1] * p_loss)

        return loss, nn_loss, p_loss

    def _get_grad(self, x, y_true, p, p_kwargs):
        """Get the gradient based on a batch of x and y_true data."""
        with tf.GradientTape() as tape:
            for layer in self._layers:
                tape.watch(layer.variables)

            y_predicted = self.predict(x, to_numpy=False)
            loss = self.loss(y_predicted, y_true, p, p_kwargs)[0]
            grad = tape.gradient(loss, self.weights)

        return grad, loss

    def _run_sgd(self, x, y_true, p, p_kwargs):
        """Run stochastic gradient descent for one batch of (x, y_true) and
        adjust NN weights."""
        grad, loss = self._get_grad(x, y_true, p, p_kwargs)
        self._optimizer.apply_gradients(zip(grad, self.weights))
        return grad, loss

    def _p_fun_preflight(self, x, y_true, p, p_kwargs):
        """Run a pre-flight check making sure the p_fun is differentiable."""

        if p_kwargs is None:
            p_kwargs = {}

        with tf.GradientTape() as tape:
            for layer in self._layers:
                tape.watch(layer.variables)

            y_predicted = self.predict(x, to_numpy=False)
            p_loss = self._p_fun(y_predicted, y_true, p, **p_kwargs)
            grad = tape.gradient(p_loss, self.weights)

            if not tf.is_tensor(p_loss):
                emsg = 'Loss output from p_fun() must be a tensor!'
                logger.error(emsg)
                raise TypeError(emsg)

            if p_loss.ndim > 1:
                emsg = ('Loss output from p_fun() should be a scalar tensor '
                        'but received a tensor with shape {}'
                        .format(p_loss.shape))
                logger.error(emsg)
                raise ValueError(emsg)

            assert isinstance(grad, list)
            if grad[0] is None:
                emsg = ('The input p_fun was not differentiable! '
                        'Please use only tensor math in the p_fun.')
                logger.error(emsg)
                raise RuntimeError(emsg)

    def _get_val_split(self, x, y, p, shuffle=True, validation_split=0.2):
        """Get a validation split and remove from from the training data.

        Parameters
        ----------
        x : np.ndarray
            Feature data in a 2D array
        y : np.ndarray
            Known output data in a 2D array.
        p : np.ndarray
            Supplemental feature data for the physics loss function in 2D array
        shuffle : bool
            Flag to randomly subset the validation data from x and y.
            shuffle=False will take the first entries in x and y.
        validation_split : float
            Fraction of x and y to put in the validation set.

        Returns
        -------
        x : np.ndarray
            Feature data for model training as 2D array. Length of this
            will be the length of the input x multiplied by one minus
            the split fraction
        y : np.ndarray
            Known output data for model training as 2D array. Length of this
            will be the length of the input y multiplied by one minus
            the split fraction
        p : np.ndarray
            Supplemental feature data for physics loss function to be used
            in model training as 2D array. Length of this will be the length
            of the input p multiplied by one minus the split fraction
        x_val : np.ndarray
            Feature data for model validation as 2D array. Length of this
            will be the length of the input x multiplied by the split fraction
        y_val : np.ndarray
            Known output data for model validation as 2D array. Length of this
            will be the length of the input y multiplied by the split fraction
        p_val : np.ndarray
            Supplemental feature data for physics loss function to be used in
            model validation as 2D array. Length of this will be the length of
            the input p multiplied by the split fraction
        """

        L = len(x)
        n = int(L * validation_split)

        if shuffle:
            i = np.random.choice(L, replace=False, size=(n,))
        else:
            i = np.arange(n)

        j = np.array(list(set(range(L)) - set(i)))

        assert len(set(i)) == len(i)
        assert len(set(list(i) + list(j))) == L

        x_val, y_val, p_val = x[i, :], y[i, :], p[i, :]
        x, y, p = x[j, :], y[j, :], p[j, :]

        self._check_shapes(x_val, y_val)
        self._check_shapes(x_val, p_val)
        self._check_shapes(x, y)
        self._check_shapes(x, p)

        logger.debug('Validation data has length {} and training data has '
                     'length {} (split of {})'
                     .format(len(x_val), len(x), validation_split))

        return x, y, p, x_val, y_val, p_val

    @staticmethod
    def _make_batches(x, y, p, n_batch=16, shuffle=True):
        """Make lists of batches from x and y.

        Parameters
        ----------
        x : np.ndarray
            Feature data for training in a 2D array
        y : np.ndarray
            Known output data for training in a 2D array.
        p : np.ndarray
            Supplemental feature data for the physics loss function in 2D array
        n_batch : int
            Number of times to update the NN weights per epoch. The training
            data will be split into this many batches and the NN will train on
            each batch, update weights, then move onto the next batch.
        shuffle : bool
            Flag to randomly subset the validation data from x and y.

        Returns
        -------
        x_batches : list
            List of 2D arrays that are split subsets of x.
            Length of list is n_batch.
        y_batches : list
            List of 2D arrays that are split subsets of y.
            Length of list is n_batch.
        p_batches : list
            List of 2D arrays that are split subsets of p.
            Length of list is n_batch.
        """

        L = len(x)
        if shuffle:
            i = np.random.choice(L, replace=False, size=(L,))
            assert len(set(i)) == L
        else:
            i = np.arange(L)

        batch_indexes = np.array_split(i, n_batch)

        x_batches = [x[j, :] for j in batch_indexes]
        y_batches = [y[j, :] for j in batch_indexes]
        p_batches = [p[j, :] for j in batch_indexes]

        return x_batches, y_batches, p_batches

    def fit(self, x, y, p, n_batch=16, epochs=10, shuffle=True,
            validation_split=0.2, p_kwargs=None):
        """Fit the neural network to data from x and y.

        Parameters
        ----------
        x : np.ndarray
            Feature data in a 2D array
        y : np.ndarray
            Known output data in a 2D array.
        p : np.ndarray
            Supplemental feature data for the physics loss function in 2D array
        n_batch : int
            Number of times to update the NN weights per epoch. The training
            data will be split into this many batches and the NN will train on
            each batch, update weights, then move onto the next batch.
        epochs : int
            Number of times to iterate on the training data.
        shuffle : bool
            Flag to randomly subset the validation data and batch selection
            from x and y.
        validation_split : float
            Fraction of x and y to use for validation.
        p_kwargs : None | dict
            Optional kwargs for the physical loss function self._p_fun.
        """

        self._check_shapes(x, y)
        self._check_shapes(x, p)

        self._history = pd.DataFrame(
            columns=['elapsed_time', 'training_loss', 'validation_loss'],
            index=np.arange(epochs))
        self._history.index.name = 'epoch'

        x, y, p, x_val, y_val, p_val = self._get_val_split(
            x, y, p, shuffle=shuffle, validation_split=validation_split)

        self._p_fun_preflight(x_val, y_val, p_val, p_kwargs)

        t0 = time.time()
        for epoch in range(epochs):
            x_batches, y_batches, p_batches = self._make_batches(
                x, y, p, n_batch=n_batch, shuffle=shuffle)

            batch_iter = zip(x_batches, y_batches, p_batches)
            for x_batch, y_batch, p_batch in batch_iter:
                grad, train_loss = self._run_sgd(x_batch, y_batch, p_batch,
                                                 p_kwargs)

            y_val_pred = self.predict(x_val, to_numpy=False)
            val_loss = self.loss(y_val_pred, y_val, p_val, p_kwargs)[0]
            logger.info('Epoch {} training loss: {:.2e} '
                        'validation loss: {:.2e}'
                        .format(epoch + 1, train_loss, val_loss))

            self._history.at[epoch, 'elapsed_time'] = time.time() - t0
            self._history.at[epoch, 'training_loss'] = train_loss.numpy()
            self._history.at[epoch, 'validation_loss'] = val_loss.numpy()

        return grad, train_loss

    def predict(self, x, to_numpy=True):
        """Run a prediction on input features.

        Parameters
        ----------
        x : np.ndarray
            Feature data in a 2D array
        to_numpy : bool
            Flag to convert output from tensor to numpy array

        Returns
        -------
        y : tf.Tensor | np.ndarray
            Predicted output data in a 2D array.
        """

        y = self._layers[0](x)
        for layer in self._layers[1:]:
            y = layer(y)

        if to_numpy:
            y = y.numpy()

        return y
