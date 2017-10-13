#!/usr/bin/env python3

import sys
from nn_smt2 import *

print(list(parse_smt2_file(sys.argv[1])))
