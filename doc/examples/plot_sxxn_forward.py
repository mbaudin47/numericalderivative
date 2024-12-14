#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Experiments with Shi, Xie, Xuan & Nocedal method for forward formula
====================================================================

References
----------
- Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.
"""

# %%
import numpy as np
import pylab as pl
import numericalderivative as nd
import sys
from matplotlib.ticker import MaxNLocator


# %%
# Use the method on a simple problem
# ----------------------------------

# %%
# In the next example, we use the algorithm on the exponential function.
# We create the :class:`~numericalderivative.DumontetVignes` algorithm using the function and the point x.
# Then we use the :meth:`~numericalderivative.DumontetVignes.compute_step()` method to compute the step,
# using an upper bound of the step as an initial point of the algorithm.
# Finally, use the :meth:`~numericalderivative.DumontetVignes.compute_first_derivative()` method to compute
# an approximate value of the first derivative using finite differences.
# The :meth:`~numericalderivative.DumontetVignes.get_number_of_function_evaluations()` method
# can be used to get the number of function evaluations.

# %%
x = 1.0
kmin = 1.0e-10
kmax = 1.0e0
algorithm = nd.DumontetVignes(np.exp, x, verbose=True)
step, number_of_iterations = algorithm.compute_step(kmin=kmin, kmax=kmax)
f_prime_approx = algorithm.compute_first_derivative(step)
feval = algorithm.get_number_of_function_evaluations()
f_prime_exact = np.exp(x)  # Since the derivative of exp is exp.
print(f"Computed step = {step:.3e}")
print(f"Number of iterations = {number_of_iterations}")
print(f"f_prime_approx = {f_prime_approx}")
print(f"f_prime_exact = {f_prime_exact}")
absolute_error = abs(f_prime_approx - f_prime_exact)
