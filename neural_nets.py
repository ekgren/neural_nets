__author__ = 'ariel'

import numpy as np
from collections import namedtuple

def sign_comp(x):
    if abs(x) < 1.:
        return 1.
    else:
        return 0.1

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
        optimal: eta = 1.0 / (t + t0) [not implemented]
        where t0 is chosen by a heuristic proposed by Leon Bottou.

    eta0 : double
        The initial learning rate for the 'constant' or 'invscaling'
        schedules. The default value is 0.5.

    """

    def __init__(self, verbose=0, learning_rate='constant', eta0=0.5):
        self.eta0 = eta0
        self.Layer = namedtuple('Layer', ['type', 'dim', 'bias', 'f'])
        self.layer_data = []
        self.layers_added = 0
        self.weights = []

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

    def transfer_func(self, h, f='step'):
        if f == 'tanh':
            return np.tanh(h)
        elif f == 'step':
            return np.sign(h)
        elif f == 'linear':
            return h

    def transfer_func_deriv(self, h, f='step'):
        pass


    def SGD_fit(self, X, Y, n_iter=1):

        for n in range(n_iter):

            E_mean = 0.

            #TODO: Randomize order of x from X
            for x, y in zip(X, Y):

                hs = []
                zs = []

                #Forward pass
                for i, layer in enumerate(self.layer_data):
                    if layer.type == 'Input':
                        if layer.bias:
                            new_x = np.append(x, 1.)
                        h = self.weights[i].dot(new_x.T)
                        hs.append(h)

                    elif layer.type == 'Hidden':
                        z = self.transfer_func(h, layer.f)
                        if layer.bias:
                            z = np.append(z, 1.)
                        zs.append(z)
                        h = self.weights[i].dot(z)
                        hs.append(h)


                    elif layer.type == 'Output':
                        o = self.transfer_func(h, layer.f)
                        zs.append(o)

                '''
                v = 10000
                dim = 100

                eta = 0.01

                w = np.zeros((1, v))
                w[:, 0] = 1

                W1 = np.random.random((v, dim))/dim
                W2 = np.random.random((dim, v))/dim

                x = w.copy()
                target = np.zeros((1, v))
                target[:, 5] = 1

                for iteration in xrange(1000):
                    h1 = np.dot(x, W1)
                    z1 = np.tanh(h)

                    h2 = np.dot(z1, W2)
                    o = np.tanh(h2)

                    df = 1. - np.power(h1, 2)
                    dg = 1. - np.power(h2, 2)

                    E = 0.5*np.power(np.linalg.norm(target - y), 2)
                    dE = o - target

                    do = dE*dg
                    dh = df*np.dot(do, W2.T)

                    #dW2 = np.outer(hidden, do)
                    dW2 = np.outer(df, do)
                    dW1 = np.outer(x, dh)

                    W2 = W2 - eta*dW2
                    W1 = W1 - eta*dW1

                    if (iteration+1)%10 == 0:
                        print "Iteration:", iteration + 1
                        print "Error:", E
                '''

                for i, (h, z) in enumerate(zip(hs[::-1], zs[::-1])):
                    if i == 0:
                        E = 0.5*np.sum(np.power(y - z, 2))
                        E_mean += E

                        dE = z - y
                        df = dS(h)
                        do = dE*df

                    elif i == 1:

                        dW = np.outer(do, dS(z))
                        self.weights[-i] = self.weights[-i] - self.eta0*dW

            E_mean = E_mean/X.shape[0]
            print E_mean

if __name__=='__main__':

    X = np.random.random((1, 100))*2-1
    y = np.random.random((1, 2))*2-1

    NN = NeuralNet(eta0=0.000001)
    NN.add_layer(layer_type='Input', n_features=100, bias=True, f=None)
    NN.add_layer(layer_type='Hidden', n_features=50, bias=True, f='step')
    NN.add_layer(layer_type='Output', n_features=2, bias=False, f='step')

    responses = NN.SGD_fit(X, y, n_iter=10)