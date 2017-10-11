#!/usr/bin/env python3

import math
import time
import sys

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S

import numpy as np
import numpy.random as R

import pylab
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    seed = np.int64(sys.argv[1])
else:
    seed = np.int64(np.float64(time.time()).view(np.uint64) % 2**32)

print("Seed:", seed)
R.seed(seed)

def random_data(size=100):
    pq = R.rand(size, 2) * 2.0 - 1.0
    pqt = pq.transpose()
    p = pqt[0]
    q = pqt[1]

    hyp = np.hypot(p, q)
    label = (hyp <= 1.0).astype(int).reshape(hyp.size, 1)

    return pq, label

def plot_classified(pq, exps, preds):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Correct")
    ax2.set_title("Predicted")
    plot_classified_impl(pq, exps, ax1)
    plot_classified_impl(pq, preds, ax2)
    pylab.show()

def plot_classified_impl(pq, label, ax):
    inds0 = np.nonzero(label == 0)
    inds1 = np.nonzero(label == 1)
    p, q = pq.transpose()
    ax.plot(p[inds0], q[inds0], 'ro')
    ax.plot(p[inds1], q[inds1], 'bo')

#if True:
if False:
    fig, ax = plt.subplots()
    ax.plot(p[inds1], q[inds1], 'ro')
    ax.plot(p[inds2], q[inds2], 'bo')
    pylab.show()

def mlp(x, hidden=[32, 32, 32], classes=2):
    hs = []
    with nn.parameter_scope('mlp'):
        h = x
        for hid, hsize in enumerate(hidden):
            with nn.parameter_scope('affine{}'.format(hid + 1)):
                aff = PF.affine(h, hsize)
                h = F.relu(aff)
                #h = F.log(1 + F.exp(aff)) # analytic function
                hs.append(h)
        with nn.parameter_scope('classifier'):
            y = PF.affine(h, classes)
    return y, hs

pq, label = random_data()

x = nn.Variable(pq.shape)
t = nn.Variable(label.shape)
y, hs = mlp(x, [8])

print (y.shape)
print (t.shape)

loss = F.mean(F.softmax_cross_entropy(y, t))

x.d, t.d = pq, label
loss.forward()
print(-1, loss.d)

#print(nn.get_parameters())

learning_rate = 1e-3
#solver = S.Sgd(learning_rate)
#solver = S.Momentum(learning_rate)
#solver = S.Nesterov(learning_rate)
solver = S.Adadelta() # 0.10 - 0.14 (2 layers: 0.04 - 0.12; 3 layers: 0.03 - 0.11)
#solver = S.Adagrad() # 0.27 - 0.33
#solver = S.RMSprop() # 0.25 - 0.28
#solver = S.Adam() # 0.19 - 0.29
#solver = S.Adamax() # 0.26 - 0.39
solver.set_parameters(nn.get_parameters())

# Train

for i in range(1000):
    x.d, t.d = random_data()
    loss.forward()
    solver.zero_grad()
    loss.backward()
    solver.weight_decay(1e-5)
    solver.update()
    if i % 100 == 0:
        print(i, loss.d)

# Show prediction

x.d, t.d = random_data()
print(t.d.reshape(100))
y.forward()
preds = y.d.argmax(axis=1)

for name, param in nn.get_parameters().items():
    print(name, param.shape, param.g.flat[:20])

#plot_classified(x.d, t.d.reshape(100), preds)

print(loss.parent)
