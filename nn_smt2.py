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
