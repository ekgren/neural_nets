__author__ = 'Ariel Ekgren'

import numpy as np
from collections import namedtuple


class NeuralNet:
    """Neural Network classifier

    Parameters
    ----------
    verbose : integer, optional
        The verbosity level

    learning_rate : string, optional
        The learning rate schedule:
        constant: eta = eta0 [default]

    eta0 : double
        The initial learning rate for the 'constant'. The default value is 0.5.

    """

    def __init__(self, verbose=0, learning_rate='constant', eta0=0.5,
                 derivative='tanh'):
        self.eta0 = eta0
        self.Layer = namedtuple('Layer', ['type', 'dim', 'bias', 'f'])
        self.layer_data = []
        self.layers_added = 0
        self.weights = []
        self.derivative = derivative

        def sign_comp(x):
            if abs(x) < 1.:
                return 1.
            else:
                return 0.1
        self.sign_comp_vectorized = np.vectorize(sign_comp)

    def add_layer(self, layer_type, n_features, bias=True, f=None):
        if layer_type != 'Input':
            in_dim = self.layer_data[self.layers_added - 1].dim
            if self.layer_data[self.layers_added - 1].bias:
                in_dim += 1
            self.weights.append(self.init_weights(n_features, in_dim))

        self.layer_data.append(self.Layer(layer_type, n_features, bias, f))
        self.layers_added += 1

    def init_weights(self, out_dim, in_dim):
        init_epsilon = 0.1
        return np.random.random((out_dim, in_dim))*2*init_epsilon-init_epsilon

    def transfer_func(self, v, f='tanh'):
        if f == 'tanh':
            return np.tanh(v)
        elif f == 'abs':
            return np.absolute(v)
        elif f == 'step':
            return np.sign(v)
        elif f == 'linear':
            return v
        elif f == 'relu':
            return np.fmax(0, v)

    def transfer_func_derivative(self, v, f='tanh'):
        if f == 'tanh':
            return 1. - np.power(np.tanh(v), 2)
        elif f == 'abs':
            return np.sign(v)
        elif f == 'step':
            return self.sign_comp_vectorized(v)
        elif f == 'linear':
            return np.ones(v.shape)
        elif f == 'relu':
            return np.fmax(0, np.sign(v))

    def error_function(self, v):
        return 0.5*np.power(np.linalg.norm(v), 2)

    def forward_pass(self, x, f=None):
        '''

        :param x:
        :return:
        '''
        activations = []
        zs = []

        for i, layer in enumerate(self.layer_data):
            if layer.type == 'Input':
                a = x
                if layer.bias:
                    a = np.append(1., a)
                activations.append(a)
                z = self.weights[i].dot(a)
                zs.append(z)

            elif layer.type == 'Hidden':
                if f:
                    a = self.transfer_func(z, f)
                else:
                    a = self.transfer_func(z, layer.f)
                if layer.bias:
                    a = np.append(1., z)
                activations.append(a)
                z = self.weights[i].dot(a)
                zs.append(z)

            elif layer.type == 'Output':
                if f:
                    a = self.transfer_func(z, f)
                else:
                    a = self.transfer_func(z, layer.f)
                activations.append(a)

        return activations, zs

    def backward_pass(self, activations, zs, y):

        deltas = []

        deltas.append(activations[-1] - y)
        self.weights[-1] = self.weights[-1] - self.eta0*np.outer(deltas[0],activations[2])

        a_prime = np.append(1., self.transfer_func_derivative(zs[-2], self.derivative))
        deltas.append(np.dot(self.weights[-1].T, deltas[-1])*a_prime)

        self.weights[-2] = self.weights[-2] - self.eta0*np.outer(deltas[1][1:],activations[1])

        a_prime = np.append(1., self.transfer_func_derivative(zs[-3], self.derivative))
        deltas.append(np.dot(self.weights[-2].T, deltas[-1][1:])*a_prime)

        self.weights[-3] = self.weights[-3] - self.eta0*np.outer(deltas[2][1:],activations[0])

        return deltas

    def SGD_fit(self, X, Y, n_iter=1):

        for n in range(n_iter):

            J_mean = 0.

            # TODO: Randomize order of x from X
            for x, y in zip(X, Y):
                activations, zs = self.forward_pass(x)

                J = self.error_function(y - activations[-1])
                J_mean += J
                ds = self.backward_pass(activations, zs, y)

            if n%40 == 0:
                J_mean = J_mean / X.shape[0]
                print J_mean

    def predict(self, X):

        Y = np.zeros((X.shape[0], self.layer_data[-1].dim))

        for i, x in enumerate(X):
            activations, zs = self.forward_pass(x)
            Y[i] = activations[-1]

        return Y


if __name__ == '__main__':
    np.random.seed(1)
    print np.random.random(2)
    X = np.random.random((10, 20)) * 2. - 1.
    Y = np.random.random((10, 3)) * 2. - 1.

    f = 'tanh'
    NN = NeuralNet(eta0=0.001, derivative='step')
    NN.add_layer(layer_type='Input', n_features=X.shape[1], bias=True, f=None)
    NN.add_layer(layer_type='Hidden', n_features=40, bias=True, f=f)
    NN.add_layer(layer_type='Hidden', n_features=20, bias=True, f=f)
    NN.add_layer(layer_type='Output', n_features=3, bias=True, f=f)

    print NN.predict(X)

    #responses = NN.SGD_fit(X, Y, n_iter=400)