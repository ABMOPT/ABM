#!/bin/sh
mpiexec -n 10 python runnerABM.py rcv1_test -1 300 0.00002117029 3e-6 10 1 1 1e-7 False                                    
mpiexec -n 10 python runnerABM.py epsilon -1 700 0.000002 5e-5 10 1 1 1e-7 False
mpiexec -n 10 python runnerABM.py MNIST8m -1 500 6.06468594e-7 3e-3 10 1 1 1e-7 False