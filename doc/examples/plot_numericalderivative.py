#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
A simple demonstration of the methods
=====================================

Finds a step which is near to optimal for a central finite difference 
formula.

References
----------
- Adaptive numerical differentiation. R. S. Stepleman and N. D. Winarsky. Journal: Math. Comp. 33 (1979), 1257-1264 
"""
# %%
import numpy as np
import pylab as pl
import numericalderivative as nd

# %%
# Define the function
# -------------------

# %%
# We first define a function.
# Here, we do not use the :class:`~numericalderivative.ScaledExponentialDerivativeBenchmark`
# class, for demonstration purposes.


# %%
def scaled_exp(x):
    alpha = 1.0e6
    return np.exp(-x / alpha)


# %%
# Define its exact derivative (for testing purposes only).
def scaled_exp_prime(x):
    alpha = 1.0e6
    return -np.exp(-x / alpha) / alpha



# %%
# We evaluate the function, its first and second derivatives at the point x.

# %%
x = 1.0e0
exact_f_value = scaled_exp(x)
print("f(x) = ", exact_f_value)
exact_f_prime_value = scaled_exp_prime(x)
print("f'(x) = ", exact_f_prime_value)

# %%
# SteplemanWinarsky
# -----------------

# %%
# In order to compute the first derivative, we use the :class:`~numericalderivative.SteplemanWinarsky`.
# This class uses the central finite difference formula.
# In order to compute a step which is approximately optimal,
# we use the :meth:`~numericalderivative.SteplemanWinarsky.find_step` method.
# Then we use the :meth:`~numericalderivative.SteplemanWinarsky.compute_first_derivative` method
# to compute the approximate first derivative and use the approximately optimal
# step as input argument.
# The input argument of :meth:`~numericalderivative.SteplemanWinarsky.find_step` is
# an upper bound of the optimal step (but this is not the case for all
# algorithms).

# %%
step_initial = 1.0e5  # An upper bound of the truly optimal step
x = 1.0e0
algorithm = nd.SteplemanWinarsky(scaled_exp, x)
step_optimal, iterations = algorithm.find_step(step_initial)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print("Optimum h =", step_optimal)
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(step_optimal)
print("f_prime_approx = ", f_prime_approx)
exact_f_prime_value = scaled_exp_prime(x)
print("exact_f_prime_value = ", exact_f_prime_value)
absolute_error = abs(f_prime_approx - exact_f_prime_value)
print(f"Absolute error = {absolute_error:.3e}")
relative_error = absolute_error / abs(exact_f_prime_value)
print(f"Relative error = {relative_error:.3e}")

# %%
# DumontetVignes
# --------------

# %%
# In the next example, we use :class:`~numericalderivative.DumontetVignes` to compute an approximately
# optimal step.
# For this algorithm, we must provide an interval which contains the
# optimal step for the approximation of the third derivative.

# %%
x = 1.0e0
algorithm = nd.DumontetVignes(scaled_exp, x)
step_optimal, _ = algorithm.find_step(
    kmin=1.0e-2,
    kmax=1.0e2,
)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print("Optimum h =", step_optimal)
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(step_optimal)
print("f_prime_approx = ", f_prime_approx)
exact_f_prime_value = scaled_exp_prime(x)
print("exact_f_prime_value = ", exact_f_prime_value)
absolute_error = abs(f_prime_approx - exact_f_prime_value)
print(f"Absolute error = {absolute_error:.3e}")
relative_error = absolute_error / abs(exact_f_prime_value)
print(f"Relative error = {relative_error:.3e}")

# %%
# GillMurraySaundersWright
# ------------------------

# %%
# In the next example, we use :class:`~numericalderivative.GillMurraySaundersWright` to compute an approximately
# optimal step.
# For this algorithm, we must provide an interval which contains the
# optimal step for the approximation of the second derivative.

# %%
x = 1.0e0
absolute_precision = 1.0e-15
algorithm = nd.GillMurraySaundersWright(scaled_exp, x, absolute_precision)
kmin = 1.0e-2
kmax = 1.0e7
step, number_of_iterations = algorithm.find_step(kmin, kmax)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print("Optimum h for f'=", step)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(step)
print("f_prime_approx = ", f_prime_approx)
exact_f_prime_value = scaled_exp_prime(x)
print("exact_f_prime_value = ", exact_f_prime_value)
absolute_error = abs(f_prime_approx - exact_f_prime_value)
print(f"Absolute error = {absolute_error:.3e}")
relative_error = absolute_error / abs(exact_f_prime_value)
print(f"Relative error = {relative_error:.3e}")


# %%
# Function with extra arguments
# -----------------------------

# %%
# Some function use extra arguments, such as parameters for examples.
# For such a function, the `args` optionnal argument can be used
# to pass extra parameters to the function.
# The goal of the :class:`~numericalderivative.FunctionWithArguments` class
# is to evaluate such a function.


# %%
# Define a function with arguments.
def my_exp_with_args(x, scaling):
    return np.exp(-x * scaling)


# %%
# Compute the derivative of a function with extra input arguments.

# %%
step_initial = 1.0e5
x = 1.0e0
scaling = 1.0e-6
algorithm = nd.SteplemanWinarsky(my_exp_with_args, x, args=[scaling])
step_optimal, iterations = algorithm.find_step(step_initial)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print("Optimum h for f''=", step_optimal)
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)

# %%
