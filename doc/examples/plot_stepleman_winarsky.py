#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Experiment with Stepleman & Winarsky method
===========================================

Find a step which is near to optimal for a centered finite difference 
formula.

References
----------
- Adaptive numerical differentiation
  R. S. Stepleman and N. D. Winarsky
  Journal: Math. Comp. 33 (1979), 1257-1264 
"""
# %%
import numpy as np
import pylab as pl
import numericalderivative as nd


# %%
# 1. Plot the error vs h
benchmark = nd.ScaledExponentialDerivativeBenchmark()
x = 1.0
finite_difference = nd.FiniteDifferenceFormula(benchmark.function, x)
number_of_points = 1000
h_array = np.logspace(-7.0, 5.0, number_of_points)
error_array = np.zeros((number_of_points))
for i in range(number_of_points):
    h = h_array[i]
    f_prime_approx = finite_difference.compute_first_derivative_central(h)
    error_array[i] = abs(f_prime_approx - benchmark.first_derivative(x))

# %%
pl.figure(figsize=(3.0, 2.0))
pl.plot(h_array, error_array)
pl.title("Finite difference")
pl.xlabel("h")
pl.ylabel("Error")
pl.xscale("log")
pl.yscale("log")


# %%
# 2. Algorithm to detect h*
algorithm = nd.SteplemanWinarsky(benchmark.function, x, verbose=True)
initial_step = 1.0e8
x = 1.0e0
h_optimal, iterations = algorithm.compute_step(initial_step)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print("Optimum h =", h_optimal)
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
absolute_error = abs(f_prime_approx - benchmark.first_derivative(x))
print("Error = ", absolute_error)


# %%
def fd_difference(h1, h2, f, x):
    finite_difference = nd.FiniteDifferenceFormula(f, x)
    f_prime_approx_1 = finite_difference.compute_first_derivative_central(h1)
    f_prime_approx_2 = finite_difference.compute_first_derivative_central(h2)
    diff_current = abs(f_prime_approx_1 - f_prime_approx_2)
    return diff_current


# %%
# 3. Plot the evolution of | FD(h) - FD(h / 2) | for different values of h
number_of_points = 1000
h_array = np.logspace(-7.0, 5.0, number_of_points)
diff_array = np.zeros((number_of_points))
for i in range(number_of_points):
    h = h_array[i]
    diff_array[i] = fd_difference(h, h / 2, benchmark.function, x)

# %%
pl.figure(figsize=(3.0, 2.0))
pl.plot(h_array, diff_array)
pl.title("F.D. difference")
pl.xlabel("h")
pl.ylabel(r"$|\operatorname{FD}(h) - \operatorname{FD}(h / 2) |$")
pl.xscale("log")
pl.yscale("log")


# %%
# 3. Plot the evolution of | FD(h) - FD(h / 2) | for different values of h
number_of_points = 20
h_initial = 1.0e5
beta = 4.0
h_array = np.zeros((number_of_points))
diff_array = np.zeros((number_of_points))
for i in range(number_of_points):
    if i == 0:
        h_array[i] = h_initial / beta
        diff_array[i] = fd_difference(h_array[i], h_initial, benchmark.function, x)
    else:
        h_array[i] = h_array[i - 1] / beta
        diff_array[i] = fd_difference(h_array[i], h_array[i - 1], benchmark.function, x)

# %%
pl.figure(figsize=(3.0, 2.0))
pl.plot(h_array, diff_array, "o")
pl.title("F.D. difference")
pl.xlabel("h")
pl.ylabel(r"$|\operatorname{FD}(h) - \operatorname{FD}(h / 2) |$")
pl.xscale("log")
pl.yscale("log")

# %%
# 4. Compute suggested step
p = 1.0e-16
beta = 4.0
h_reference = beta * p ** (1 / 3) * x
print("Suggested h0 = ", h_reference)

# %%
# 5. Plot number of lost digits vs h
h = 1.0e4
print("Starting h = ", h)
n_digits = algorithm.number_of_lost_digits(h)
print("Number of lost digits = ", n_digits)
threshold = np.log10(p ** (-1.0 / 3.0) / beta)
print("Threshold = ", threshold)

bracket_step = [1.0e-5, 1.0e7]
step_zero, iterations = algorithm.search_step_with_bisection(
    bracket_step,
)
print("step_zero = ", step_zero)
print("iterations = ", iterations)

estim_step, iterations = algorithm.compute_step(step_zero, beta=1.5)
print("estim_step = ", estim_step)
print("iterations = ", iterations)

# %%
number_of_points = 1000
h_array = np.logspace(-7.0, 7.0, number_of_points)
n_digits_array = np.zeros((number_of_points))
for i in range(number_of_points):
    h = h_array[i]
    n_digits_array[i] = algorithm.number_of_lost_digits(h)

# %%
y_max = algorithm.number_of_lost_digits(h_reference)
pl.figure(figsize=(3.0, 2.0))
pl.plot(h_array, n_digits_array, label="$N(h)$")
pl.plot([h_reference] * 2, [0.0, y_max], "--", label=r"$h_{ref}$")
pl.plot([step_zero] * 2, [0.0, y_max], "--", label=r"$h^{(0)}$")
pl.plot([estim_step] * 2, [0.0, y_max], "--", label=r"$h^\star$")
pl.plot(
    h_array,
    np.array([threshold] * number_of_points),
    ":",
    label=r"$T$",
)
pl.title("Number of digits lost by F.D.")
pl.xlabel("h")
pl.ylabel("$N(h)$")
pl.xscale("log")
_ = pl.legend(bbox_to_anchor=(1.1, 1.0))


# %%
pl.figure(figsize=(3.0, 2.0))
pl.plot(h_array, error_array)
pl.plot([step_zero] * 2, [0.0, 1.0e-9], "--", label=r"$h^{(0)}$")
pl.plot([estim_step] * 2, [0.0, 1.0e-9], "--", label=r"$h^\star$")
pl.title("Finite difference")
pl.xlabel("h")
pl.ylabel("Error")
pl.xscale("log")
pl.legend(bbox_to_anchor=(1.1, 1.0))
pl.yscale("log")

# %%
# 6. Benchmark
# Test with single point
x = 1.0
bracket_step = [1.0e-7, 1.0e7]
f_prime_approx, number_of_iterations = algorithm.search_step_with_bisection(
    bracket_step,
)
feval = algorithm.number_of_function_evaluations
print("FD(x) = ", f_prime_approx)
print("number_of_iterations = ", number_of_iterations)
print("Func. eval = ", feval)

# %%
# Algorithme de dichotomie pour le pas initial
bracket_step = [1.0e-7, 1.0e1]
x = 1.0
maximum_bisection = 53
log_scale = False
h0, iterations = algorithm.search_step_with_bisection(
    bracket_step,
    maximum_bisection=53,
    log_scale=False,
)
print("Pas initial = ", h0, ", iterations = ", iterations)
h0, iterations = algorithm.search_step_with_bisection(
    bracket_step,
    maximum_bisection=53,
    log_scale=True,
)
print("Pas initial = ", h0, ", iterations = ", iterations)

# %%
# Test
benchmark = nd.ExponentialDerivativeBenchmark()
bracket_step = [1.0e-6, 100.0 * x]
x = 1.0
algorithm = nd.SteplemanWinarsky(benchmark.function, x, verbose=True)
f_prime_approx, estim_relative_error = algorithm.search_step_with_bisection(
    bracket_step,
    beta=4.0,
)
absolute_error = abs(f_prime_approx - benchmark.first_derivative(x))
feval = algorithm.number_of_function_evaluations
print(
    "x = %.3f, abs. error = %.3e, estim. rel. error = %.3e, Func. eval. = %d"
    % (x, absolute_error, estim_relative_error, number_of_function_evaluations)
)

#
