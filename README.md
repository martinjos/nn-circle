# nnabla-circle

This program uses [NNabla] to train a neural network classifier using the
[ReLU] activation function on a problem that uses the unit circle as the
decision boundary.  The trained neural network is then exported in [SMT-LIBv2]
format to allow formal verification of its properties.

The primary purpose of this is to facilitate the testing of formal verification
methods for ReLU neural network classifiers.

[NNabla]: https://nnabla.org/
[ReLU]: https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
[SMT-LIBv2]: http://www.smtlib.org/
