import math
import time
import sys

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
import nnabla.solvers as S
import nnabla.utils.save as nn_save

import numpy as np
import numpy.random as R

import pylab
import matplotlib.pyplot as plt

BATCH_SIZE = 100

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def random_data(size=BATCH_SIZE):
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

def setup_network():
    pq, label = random_data()

    x = nn.Variable(pq.shape)
    t = nn.Variable(label.shape)
    y, hs = mlp(x, [8])

    eprint (y.shape)
    eprint (t.shape)

    loss = F.mean(F.softmax_cross_entropy(y, t))

    return x, t, y, loss, hs

def train_network(loss, x, t):
    solver = S.Adadelta() # 0.10 - 0.14 (2 layers: 0.04 - 0.12; 3 layers: 0.03 - 0.11)
    solver.set_parameters(nn.get_parameters())

    for i in range(1000):
        x.d, t.d = random_data()
        loss.forward()
        solver.zero_grad()
        loss.backward()
        solver.weight_decay(1e-5)
        solver.update()
        if i % 100 == 0:
            eprint(i, loss.d)

    return solver

def predict(pq, label, x, t, y, loss):
    x.d, t.d = pq, label
    eprint(t.d.reshape(BATCH_SIZE))
    loss.forward()
    preds = y.d.argmax(axis=1)
    return preds, loss

def seed():
    if len(sys.argv) > 1:
        seed = np.int64(sys.argv[1])
    else:
        seed = np.int64(np.float64(time.time()).view(np.uint64) % 2**32)

    eprint("Seed:", seed)
    R.seed(seed)

def nnabla_to_smt2_info(var, names={}, collect={}, rcollect={}, vars=[],
                        assertions=[], nid=0, normal=True):

    if var in rcollect:
        return collect  # already processed this variable
    rcollect[var] = nid
    if var not in names:
        names[var] = 'var_{}'.format(nid)
    collect[nid] = var
    cur_name = names[var]
    if normal:
        assert len(var.shape) == 2
        for index in range(var.shape[1]):
            vars.append('{}_{}'.format(cur_name, index))
    nid += 1
    if var.parent is not None:
        eprint(var.parent)
        eprint(var.parent.inputs)
        eprint(type(var.parent.inputs))
        for index, input in enumerate(var.parent.inputs):
            _, _, _, nid = nnabla_to_smt2_info(input, names, collect, rcollect,
                                               vars, assertions, nid, index == 0)

        if var.parent.name == 'ReLU':
            assert normal
            assert len(var.parent.inputs) == 1
            assert var.parent.inputs[0].shape == var.shape
            param_name = names[var.parent.inputs[0]]
            for index in range(var.shape[1]):
                assertions.append('(= {}_{} (max 0 {}_{}))'.format(
                    cur_name, index, param_name, index
                ))
        elif var.parent.name == 'Affine':
            # Wx + b -- W and b are trained parameters
            assert normal
            assert len(var.parent.inputs) == 3
            var_x = var.parent.inputs[0]
            var_W = var.parent.inputs[1]
            var_b = var.parent.inputs[2]
            assert len(var_x.shape) == 2
            assert len(var_W.shape) == 2
            assert len(var_b.shape) == 1
            assert var_W.shape[0] == var_x.shape[1]
            assert var_W.shape[1] == var.shape[1]
            assert var_W.shape[1] == var_b.shape[0]
            x_name = names[var_x]
            for i in range(var.shape[1]):
                terms = []
                for j in range(var_x.shape[1]):
                    terms.append('(* {} {}_{})'.format(
                        var_W.d[j][i],
                        x_name,
                        j
                    ))
                assertions.append('(= {}_{} (+ {} {}))'.format(
                    cur_name,
                    i,
                    var_b.d[i],
                    ' '.join(terms)
                ))
        else:
            raise Exception('Unsupported function: {}'.format(var.parent.name))
    return collect, vars, assertions, nid

def nnabla_to_smt2(var, names={}):
    collect, vars, assertions, _ = nnabla_to_smt2_info(var, names)
    smt2 = ''
    smt2 += '(set-logic QF_NRA)\n'
    smt2 += ''.join(map(lambda n: '(declare-fun {} () Real)\n'.format(n), vars))
    smt2 += ''.join(map(lambda a: '(assert {})\n'.format(a), assertions))
    smt2 += '(check-sat)\n'
    smt2 += '(exit)\n'
    return smt2
