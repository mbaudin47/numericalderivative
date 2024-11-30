#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
A simple demonstration of the methods
=====================================

Finds a step which is near to optimal for a centered finite difference 
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
# Define a function
# Here, we do not use the ScaledExponentialDerivativeBenchmark class, for demonstration purposes
def my_scaled_exp(x):
    alpha = 1.0e6
    return np.exp(-x / alpha)


# %%
# Define its exact derivative (for testing purposes only)
def my_scaled_exp_prime(x):
    alpha = 1.0e6
    return -np.exp(-x / alpha) / alpha


# %%
# Define its exact second derivative (for testing purposes only)
def my_scaled_exp_second(x):
    alpha = 1.0e6
    return np.exp(-x / alpha) / alpha**2


# %%
# Function value
print("+ Function")
x = 1.0e0
exact_f_value = my_scaled_exp(x)
print("exact_f_value = ", exact_f_value)
exact_f_prime_value = my_scaled_exp_prime(x)
print("exact_f_prime_value = ", exact_f_prime_value)
exact_f_second_value = my_scaled_exp_second(x)
print("exact_f_second_value = ", exact_f_second_value)

# %%
# Algorithm to detect h*: SteplemanWinarsky
print("+ SteplemanWinarsky")
h0 = 1.0e5
x = 1.0e0
algorithm = nd.SteplemanWinarsky(my_scaled_exp, x)
h_optimal, iterations = algorithm.compute_step(h0)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print("Optimum h =", h_optimal)
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
print("f_prime_approx = ", f_prime_approx)
exact_f_prime_value = my_scaled_exp_prime(x)
print("exact_f_prime_value = ", exact_f_prime_value)
absolute_error = abs(f_prime_approx - exact_f_prime_value)
print(f"Absolute error = {absolute_error:.3e}")
relative_error = absolute_error / abs(exact_f_prime_value)
print(f"Relative error = {relative_error:.3e}")

# %%
# Algorithm to detect h*: DumontetVignes
print("+ DumontetVignes")
x = 1.0e0
algorithm = nd.DumontetVignes(my_scaled_exp, x)
h_optimal, _ = algorithm.compute_step(
    kmin=1.0e-2,
    kmax=1.0e2,
)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print("Optimum h =", h_optimal)
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
print("f_prime_approx = ", f_prime_approx)
exact_f_prime_value = my_scaled_exp_prime(x)
print("exact_f_prime_value = ", exact_f_prime_value)
absolute_error = abs(f_prime_approx - exact_f_prime_value)
print(f"Absolute error = {absolute_error:.3e}")
relative_error = absolute_error / abs(exact_f_prime_value)
print(f"Relative error = {relative_error:.3e}")

# %%
# Algorithm to detect h*: GillMurraySaundersWright
print("+ GillMurraySaundersWright")
x = 1.0e0
absolute_precision = 1.0e-15
algorithm = nd.GillMurraySaundersWright(my_scaled_exp, x, absolute_precision)
kmin = 1.0e-2
kmax = 1.0e7
step, number_of_iterations = algorithm.compute_step(kmin, kmax)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print("Optimum h for f'=", step)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(step)
print("f_prime_approx = ", f_prime_approx)
exact_f_prime_value = my_scaled_exp_prime(x)
print("exact_f_prime_value = ", exact_f_prime_value)
absolute_error = abs(f_prime_approx - exact_f_prime_value)
print(f"Absolute error = {absolute_error:.3e}")
relative_error = absolute_error / abs(exact_f_prime_value)
print(f"Relative error = {relative_error:.3e}")


# %%
# Define a function with arguments
def my_exp_with_args(x, scaling):
    return np.exp(-x / scaling)


# %%
# Compute the derivative of a function with extra input arguments
print("+ Function with extra input arguments")
h0 = 1.0e5
x = 1.0e0
scaling = 1.0e6
algorithm = nd.SteplemanWinarsky(my_exp_with_args, x, args=[scaling])
h_optimal, iterations = algorithm.compute_step(h0)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print("Optimum h for f''=", h_optimal)
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)

# %%
