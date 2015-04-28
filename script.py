__author__ = 'Ariel Ekgren'

from neural_nets import NeuralNet
import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import load_iris


np.random.seed(2)
print np.random.random(2)


digits = load_digits()
print digits.data.shape
print digits.target.shape
target = np.zeros((1797, 10))
for i, no in enumerate(digits.target):
    target[i, no] = 1

X = digits.data[:1000]
Y = target[:1000]

'''
iris = load_iris()
print iris.data.shape
print iris.target.shape

target = np.zeros((150, 3))
for i, no in enumerate(iris.target):
    target[i, no] = 1

X = iris.data
Y = target
'''
f = 'relu'
NN = NeuralNet(eta0=0.0001, derivative='relu')
NN.add_layer(layer_type='Input', n_features=X.shape[1], bias=True, f=None)
NN.add_layer(layer_type='Hidden', n_features=100, bias=True, f=f)
NN.add_layer(layer_type='Hidden', n_features=16, bias=True, f=f)
NN.add_layer(layer_type='Output', n_features=Y.shape[1], bias=True, f=f)

responses = NN.SGD_fit(X, Y, n_iter=401)