#!/bin/sh
mpiexec -n 10 python runnerDAVERPG.py rcv1_test -1 300 0.00002117029 3e-6 142.85                                    
mpiexec -n 10 python runnerDAVERPG.py epsilon -1 700 0.000002 5e-5 11.36
mpiexec -n 10 python runnerDAVERPG.py MNIST8m -1 500 6.06468594e-7 3e-3 0.106