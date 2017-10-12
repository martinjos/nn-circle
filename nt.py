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

#for name, param in nn.get_parameters().items():
#    print(name, param.shape, param.g.flat[:20])

#print("Test loss:", loss.d)

#plot_classified(x.d, t.d.reshape(BATCH_SIZE), preds)

smt2 = nnabla_to_smt2(y, {x: 'x', y: 'y'})
print(smt2)
