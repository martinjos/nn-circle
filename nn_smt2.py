import re
from pathlib import Path

from shared import *

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
                assertions.append('(= {}_{} (max 0.0 {}_{}))'.format(
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
                    terms.append('(* {: .17f} {}_{})'.format(
                        var_W.d[j][i],
                        x_name,
                        j
                    ))
                assertions.append('(= {}_{} (+ {: .17f} {}))'.format(
                    cur_name,
                    i,
                    var_b.d[i],
                    ' '.join(terms)
                ))
        else:
            raise Exception('Unsupported function: {}'.format(var.parent.name))
    return collect, vars, assertions, nid

def nnabla_to_smt2(var, names={}, save_test=None, seed=None, test_seed=None,
                   test_eps=1e-6, test_batch=None, include=None, std=False):
    collect, vars, assertions, _ = nnabla_to_smt2_info(var, names)
    smt2 = ''
    smt2 += '(set-logic QF_NRA)\n'
    if std:
        smt2 += '\n(define-fun max ((x Real) (y Real)) Real ' \
                '(ite (>= x y) x y))\n'
    if seed:
        smt2 += '\n; Training seed = {}\n'.format(seed)
    smt2 += '\n; NN variables\n\n'
    smt2 += ''.join(map(lambda n: '(declare-fun {} () Real)\n'.format(n), vars))
    smt2 += '\n; NN assertions\n\n'
    smt2 += ''.join(map(lambda a: '(assert {})\n'.format(a), assertions))
    smt2 += '\n'
    if save_test is not None:
        if test_seed:
            smt2 += '; Test seed = {}\n\n'.format(test_seed)
        (x, y) = (save_test, var)
        assert x.shape[0] == y.shape[0]
        assert test_batch <= x.shape[0]
        smt2 += '; Assertion for test data\n\n'
        cases = []
        for i in range(0, test_batch):
            cases.append(('(and (= {} {: .17f}) (= {} {: .17f})\n '
                          ' (or (< {} {: .17f}) (> {} {: .17f})\n '
                          '     (< {} {: .17f}) (> {} {: .17f})))').format(
                names[x] + '_0', x.d[i][0],
                names[x] + '_1', x.d[i][1],
                names[y] + '_0', y.d[i][0] - test_eps,
                names[y] + '_0', y.d[i][0] + test_eps,
                names[y] + '_1', y.d[i][1] - test_eps,
                names[y] + '_1', y.d[i][1] + test_eps
            ))
        smt2 += '(assert (or\n {}))\n\n'.format('\n '.join(cases))
    if include is not None:
        smt2 += Path(include).read_text('utf-8') + '\n'
    smt2 += '(check-sat)\n'
    smt2 += '(exit)\n'
    return smt2

def parse_smt2(string):
    tokreg = r'(?x) ( \s+ )| [()] | [^\s()]+ '
    cur = []
    stack = []
    for match in re.finditer(tokreg, string):
        if match.group(1) is not None:
            continue
        elif match.group(0) == '(':
            new = []
            stack.append(cur)
            cur.append(new)
            cur = new
        elif match.group(0) == ')':
            cur = stack.pop()
        else:
            try:
                cur.append(float(match.group(0)))
            except ValueError:
                cur.append(match.group(0))
    return cur

def parse_smt2_file(filename):
    return parse_smt2(Path(filename).read_text('utf-8'))
