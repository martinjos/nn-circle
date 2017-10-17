# nn-circle

This project uses [NNabla] to train a neural network (NN) classifier using the
[ReLU] activation function on a problem that uses the unit circle as the
decision boundary.  The trained neural network is then exported in [SMT-LIBv2]
format to allow formal verification of its properties.

The primary purpose of this is to facilitate the testing of formal verification
methods for ReLU neural network classifiers.

[NNabla]: https://nnabla.org/
[ReLU]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
[SMT-LIBv2]: http://www.smtlib.org/

## Getting started

Just the NN (always SAT):

```
./nntrain.py > output.smt2
```

Test smt2-encoded NN function for deviations from known NN output
values (want UNSAT):

```
./nntrain.py --save-tests=10 > output_tests.smt2
```

Verify that the NN will never misclassify any points outside a certain
neighborhood of the decision boundary (want UNSAT):

```
./nntrain.py --include=examples/radius_test.smt2 > output_mc.smt2
```

Further options:

```
./nntrain.py -h
```
