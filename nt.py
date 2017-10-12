#!/usr/bin/env python3

import nnabla as nn
import nnabla.functions as F  # it crashes without this
import numpy.random as R
import itertools as IT

# I hate you, Python
import os
from pathlib import Path
exec(Path(os.path.dirname(__file__) + '/' + 'ntshared.py').read_text())

seed()

x, t, y, loss, hs = setup_network()

train_network(loss)

R.seed() # reseed for test data

pq, label = random_data()
preds, loss = predict(pq, label)

for name, param in nn.get_parameters().items():
    print(name, param.shape, param.g.flat[:20])

print("Test loss:", loss.d)

#plot_classified(x.d, t.d.reshape(BATCH_SIZE), preds)

# Yet another reason to hate Python: no Array#flatten !
def flatten(items, seqtypes=(list, tuple)):
    for i, x in enumerate(items):
        while i < len(items) and isinstance(items[i], seqtypes):
            if isinstance(items, tuple):
                items = list(items)
            items[i:i+1] = items[i]
    return items

def nnabla_to_smt2(var, collect={}, rcollect={}, assertions=[],
                   nid=0, normal=True):
    if var in rcollect:
        return collect  # already processed this variable
    rcollect[var] = nid
    collect[nid] = var
    cur_nid = nid
    nid += 1
    if var.parent is not None:
        print(var.parent)
        print(var.parent.inputs)
        print(type(var.parent.inputs))
        for index, input in enumerate(var.parent.inputs):
            _, _, nid = nnabla_to_smt2(input, collect, rcollect, assertions,
                                          nid, index == 0)
        if var.parent.name == 'ReLU':
            assert normal
            assert len(var.parent.inputs) == 1
            assert var.parent.inputs[0].shape == var.shape
            param_nid = rcollect[var.parent.inputs[0]]
            r = range(var.shape[1])
            # make multi-dimensional index iterator for all but first dim
            for i in range(2, len(var.shape)):
                r = IT.product(r, range(var.shape[i]))
            for index in r:
                # flatten multi-dimensional index into sequence of ints
                if not isinstance(index, tuple):
                    index = (index,)
                index = flatten(index)
                index_str = '_'.join(map(str, index))
                assertions.append('(= var_{}_{} (max 0 var_{}_{}))'.format(
                    cur_nid, index_str, param_nid, index_str
                ))
        elif var.parent.name == 'Affine':
            # Wx + b -- W and b are trained parameters
            assert normal
            assert len(var.shape) == 2
            assert len(var.parent.inputs) == 3
            var_x = var.parent.inputs[0]
            var_W = var.parent.inputs[1]
            var_b = var.parent.inputs[2]
            assert len(var_x.shape) == 2
            assert len(var_W.shape) == 2
            assert len(var_b.shape) == 1
        else:
            raise Exception('Unsupported function: {}'.format(var.parent.name))
    return collect, assertions, nid

collect, assertions, _ = nnabla_to_smt2(y)
for nid, var in collect.items():
    print(nid, var, var.ndim)
for assertion in assertions:
    print(assertion)
