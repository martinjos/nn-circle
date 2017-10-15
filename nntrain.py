#!/usr/bin/env python3

import argparse

import nnabla as nn
import nnabla.functions as F  # it crashes without this
import numpy.random as R
import itertools as IT

from nn_circle import *
from nn_smt2 import *
from shared import *

parser = argparse.ArgumentParser(description='Generate ReLU neural network for unit circle classifier.')
parser.add_argument('-s', '--seed', type=int,
                    help='random seed for training phase')
parser.add_argument('-t', '--test-seed', type=int,
                    help='random seed for test phase')
parser.add_argument('-L', '--layers', type=int, default=1,
                    help='number of hidden layers of neural network')
parser.add_argument('-S', '--size', type=int, default=8,
                    help='size of each hidden layer of neural network')
parser.add_argument('--plot', action='store_true',
                    help='plot test results')
args = parser.parse_args()

seed(args.seed)

x, t, y, loss, hs = setup_network(args.layers, args.size)

train_network(loss, x, t)

seed(args.test_seed) # reseed for test data

pq, label = random_data()
preds, loss = predict(pq, label, x, t, y, loss)

#for name, param in nn.get_parameters().items():
#    print(name, param.shape, param.g.flat[:20])

eprint("Test loss:", loss.d)

smt2 = nnabla_to_smt2(y, {x: 'x', y: 'y'})
print(smt2)

if args.plot:
    plot_classified(x.d, t.d.reshape(BATCH_SIZE), preds)
