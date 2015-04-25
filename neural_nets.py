__author__ = 'ariel'

import numpy as np
from collections import namedtuple

def sign_comp(x):
    if abs(x) < 1.:
        return 1.
    else:
        return 0.5

dS = np.vectorize(sign_comp)

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

    def __init__(self, verbose=0, learning_rate='constant', eta0=0.5):
        self.eta0 = eta0
        self.Layer = namedtuple('Layer', ['type', 'dim', 'bias', 'f'])
        self.layer_data = []
        self.layers_added = 0
        self.weights = []

        def sign_comp(x):
            if abs(x) < 1.:
                return 1.
            else:
                return 0.5
        self.sign_comp_vectorized = np.vectorize(sign_comp)

    def add_layer(self, layer_type, n_features, bias=True, f=None):
        if layer_type != 'Input':
            in_dim = self.layer_data[self.layers_added-1].dim
            if self.layer_data[self.layers_added-1].bias:
                in_dim += 1
            self.weights.append(self.init_weights(n_features, in_dim))

        self.layer_data.append(self.Layer(layer_type, n_features, bias, f))
        self.layers_added += 1

    def init_weights(self, out_dim, in_dim):
        return np.random.random((out_dim, in_dim))*2-1

    def transfer_func(self, v, f='tanh'):
        if f == 'tanh':
            return np.tanh(v)
        elif f == 'step':
            return np.sign(v)
        elif f == 'linear':
            return v

    def transfer_func_deriv(self, v, f='tanh'):
        if f == 'tanh':
            return 1. - np.power(np.tanh(v), 2)
        elif f == 'step':
            return self.sign_comp_vectorized(v)
        elif f == 'linear':
            return 1

    def error_function(self, v):
        return 0.5*np.sum(np.power(v, 2))

    def forward_pass(self, x):
        '''

        :param x:
        :return:
        '''
        activations = []
        zs = []

        for i, layer in enumerate(self.layer_data):
            if layer.type == 'Input':
                if layer.bias:
                    x = np.append(1., x)
                activations.append(x)
                z = self.weights[i].dot(x.T)
                zs.append(z)

            elif layer.type == 'Hidden':
                a = self.transfer_func(z, layer.f)
                if layer.bias:
                    a = np.append(1., z)
                activations.append(a)
                z = self.weights[i].dot(a)
                zs.append(z)

            elif layer.type == 'Output':
                a = self.transfer_func(z, layer.f)
                activations.append(a)

        return activations, zs

    def SGD_fit(self, X, Y, n_iter=1):

        for n in range(n_iter):

            J_mean = 0.

            #TODO: Randomize order of x from X
            for x, y in zip(X, Y):

                activations, zs = self.forward_pass(x)
                J = self.error_function(y - activations[-1])
                J_mean += J

                ds = []
                #Backward pass
                for i, (a, z) in enumerate(zip(activations[::-1], zs[::-1])):
                    if i == 0:
                        d = a-y
                    else:
                        d = np.dot(self.weights[-i].T, d)*dS(z)
                    ds.append(d)

                ds = ds[::-1]
                for i, w in enumerate(self.weights):
                    self.weights[i] = self.weights[i] + np.outer(ds[i], a[i])

            J_mean = J_mean/X.shape[0]
            print J_mean

if __name__=='__main__':

    X = np.random.random((1, 10))*2.-1.
    Y = np.zeros((1, 3))
    Y[0] = 1.

    NN = NeuralNet(eta0=0.0001)
    NN.add_layer(layer_type='Input', n_features=X.shape[1], bias=False, f=None)
    NN.add_layer(layer_type='Hidden', n_features=150, bias=False, f='step')
    NN.add_layer(layer_type='Hidden', n_features=50, bias=False, f='step')
    NN.add_layer(layer_type='Output', n_features=3, bias=False, f='step')

    responses = NN.SGD_fit(X, Y, n_iter=10)