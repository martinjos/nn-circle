# nn-circle

This project uses [NNabla] to train a neural network classifier using the
[ReLU] activation function on a problem that uses the unit circle as the
decision boundary.  The trained neural network is then exported in [SMT-LIBv2]
format to allow formal verification of its properties.

The primary purpose of this is to facilitate the testing of formal verification
methods for ReLU neural network classifiers.

[NNabla]: https://nnabla.org/
[ReLU]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
[SMT-LIBv2]: http://www.smtlib.org/

## Getting started

```
./nntrain.py > output.smt2  # just the NN
./nntrain.py --save-tests=10 > output_tests.smt2  # test against known outputs
./nntrain.py --include=examples/radius_test.smt2 > output_mc.smt2  # find misclassified points
./nntrain.py -h  # further options
```
