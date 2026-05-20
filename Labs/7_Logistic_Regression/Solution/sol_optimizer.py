############################################################################################
# Copyright (C) 2024 by Sandro Cumani                                                      #
#                                                                                          #
# This file is provided for didactic purposes only, according to the Politecnico di Torino #
# policies on didactic material.                                                           #
#                                                                                          #
# Any form of re-distribution or online publication is forbidden.                          #
#                                                                                          #
# This file is provided as-is, without any warranty                                        #
############################################################################################

import numpy
import scipy.optimize

def f(x):
    y,z = x
    return (y+3)**2 + numpy.sin(y) + (z+1)**2

def fprime(x):
    y,z = x
    return numpy.array([2*(y+3) + numpy.cos(y), 2 * (z+1)])

print (scipy.optimize.fmin_l_bfgs_b(func = f, approx_grad = True, x0 = numpy.zeros(2)))
print (scipy.optimize.fmin_l_bfgs_b(func = f, fprime = fprime, x0 = numpy.zeros(2)))
       

