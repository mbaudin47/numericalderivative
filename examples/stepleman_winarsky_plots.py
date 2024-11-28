#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Experiment with Stepleman & Winarsky method. 
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
import tabulate
import numericalderivative as nd


# %%
# Plot the number of lost digits for exp
number_of_points = 100
x = 1.0
h_array = np.logspace(-15.0, 1.0, number_of_points)
n_digits_array = np.zeros((number_of_points))
algorithm = nd.SteplemanWinarsky(np.exp, x)
for i in range(number_of_points):
    h = h_array[i]
    n_digits_array[i] = algorithm.number_of_lost_digits(h)

pl.figure(figsize=(3.0, 2.0))
pl.plot(h_array, n_digits_array)
pl.title(r"Number of digits lost by F.D.. $f(x) = \exp(x)$")
pl.xlabel("h")
pl.ylabel("$N(h)$")
pl.xscale("log")

# %%
# Plot the number of lost digits for sin
x = 1.0
h_array = np.logspace(-7.0, 2.0, number_of_points)
n_digits_array = np.zeros((number_of_points))
algorithm = nd.SteplemanWinarsky(np.sin, x)
for i in range(number_of_points):
    h = h_array[i]
    n_digits_array[i] = algorithm.number_of_lost_digits(h)

# %%
pl.figure(figsize=(3.0, 2.0))
pl.plot(h_array, n_digits_array)
pl.title(r"Number of digits lost by F.D.. $f(x) = \sin(x)$")
pl.xlabel("h")
pl.ylabel("$N(h)$")
pl.xscale("log")

# For each function, at point x = 1, plot the error vs the step computed
# by the method


# %%
def plot_error_vs_h_with_SW_steps(
    name, function, function_derivative, x, h_array, bracket_step, verbose=False
):
    algorithm = nd.SteplemanWinarsky(function, x)
    number_of_points = len(h_array)
    error_array = np.zeros((number_of_points))
    for i in range(number_of_points):
        h = h_array[i]
        f_prime_approx = algorithm.compute_first_derivative(h)
        error_array[i] = abs(f_prime_approx - function_derivative(x))

    bisection_h0_step, bisection_h0_iteration = algorithm.search_step_with_bisection(
        bracket_step
    )
    bisection_step, bisection_iterations = algorithm.compute_step(bisection_h0_step)
    zero_h0_step, zero_h0_iteration = algorithm.search_step_with_bisection(bracket_step)
    zero_step, zero_iteration = algorithm.compute_step(zero_h0_step)

    if verbose:
        print(name)
        print(
            "Bisection h0 = %.3e using %d iterations"
            % (bisection_h0_step, bisection_h0_iteration)
        )
        print(
            "Bisection h* = %.3e using %d iterations"
            % (bisection_step, bisection_iterations)
        )
        print("Zero h0 = %.3e using %d iterations" % (zero_h0_step, zero_h0_iteration))
        print("Zero h* = %.3e using %d iterations" % (zero_step, zero_iteration))

    minimum_error = np.nanmin(error_array)
    maximum_error = np.nanmax(error_array)

    pl.figure(figsize=(3.0, 2.0))
    pl.plot(h_array, error_array)
    pl.plot(
        [bisection_h0_step] * 2,
        [minimum_error, maximum_error],
        "--",
        label="$h_{0}^{(B)}$",
    )
    pl.plot(
        [zero_h0_step] * 2, [minimum_error, maximum_error], ":", label="$h_0^{(Z)}$"
    )
    pl.plot(
        [bisection_step] * 2, [minimum_error, maximum_error], "--", label="$h^{(B)}$"
    )
    pl.plot([zero_step] * 2, [minimum_error, maximum_error], ":", label="$h^{(Z)}$")
    pl.title("Finite difference : %s at point x = %.0f" % (name, x))
    pl.xlabel("h")
    pl.ylabel("Error")
    pl.xscale("log")
    pl.yscale("log")
    pl.legend(bbox_to_anchor=(1.0, 1.0))
    return


# %%
def plot_error_vs_h_benchmark(benchmark, x, h_array, bracket_step, verbose=False):
    plot_error_vs_h_with_SW_steps(
        benchmark.name,
        benchmark.function,
        benchmark.first_derivative,
        x,
        h_array,
        bracket_step,
        True,
    )


# %%
benchmark = nd.ExponentialDerivativeBenchmark()
x = 1.0
number_of_points = 1000
h_array = np.logspace(-15.0, 1.0, number_of_points)
bracket_step = [1.0e-10, 1.0e0]
plot_error_vs_h_benchmark(benchmark, x, h_array, bracket_step, True)

# %%
x = 12.0
h_array = np.logspace(-15.0, 1.0, number_of_points)
plot_error_vs_h_benchmark(benchmark, x, h_array, bracket_step)

if False:
    benchmark = nd.LogarithmicDerivativeBenchmark()
    x = 1.0
    bracket_step = [1.0e-15, 1.0e0]
    plot_error_vs_h_benchmark(benchmark, x, h_array, bracket_step, True)

# %%
benchmark = nd.LogarithmicDerivativeBenchmark()
x = 1.1
bracket_step = [1.0e-14, 1.0e-4]
h_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(benchmark, x, h_array, bracket_step, True)

# %%
benchmark = nd.SinDerivativeBenchmark()
x = 1.0
bracket_step = [1.0e-15, 1.0e-3]
h_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(benchmark, x, h_array, bracket_step)

# %%
benchmark = nd.SquareRootDerivativeBenchmark()
x = 1.0
bracket_step = [1.0e-15, 1.0e-1]
h_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(benchmark, x, h_array, bracket_step, True)

# %%
benchmark = nd.AtanDerivativeBenchmark()
x = 1.0
bracket_step = [1.0e-15, 1.0e-2]
h_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(benchmark, x, h_array, bracket_step)

# %%
benchmark = nd.ExponentialDerivativeBenchmark()
print("+ Sensitivity of SW step depending on h0")
print("Case 1 : exp")
x = 1.0
algorithm = nd.SteplemanWinarsky(
    benchmark.function,
    x,
)
finite_difference_optimal_step = nd.FiniteDifferenceOptimalStep()
third_derivative_value = benchmark.third_derivative(benchmark.x)
optimal_step, absolute_error = (
    finite_difference_optimal_step.compute_step_first_derivative_central(
        third_derivative_value
    )
)
print("Exact h* = %.3e" % (optimal_step))
print("absolute_error = %.3e" % (absolute_error))
for h0 in np.logspace(-4, 0, 10):
    estim_step, iterations = algorithm.compute_step(h0)
    print("h0 = %.3e, Approx. h* = %.3e (%d iterations)" % (h0, estim_step, iterations))

print("Case 2 : Scaled exp")
x = 1.0

# %%
benchmark = nd.ScaledExponentialDerivativeBenchmark()
algorithm = nd.SteplemanWinarsky(benchmark.function, x)
finite_difference_optimal_step = nd.FiniteDifferenceOptimalStep()
third_derivative_value = benchmark.third_derivative(benchmark.x)
optimal_step, absolute_error = (
    finite_difference_optimal_step.compute_step_first_derivative_central(
        third_derivative_value
    )
)
print("Exact h* = %.3e" % (optimal_step))
print("absolute_error = %.3e" % (absolute_error))
for h0 in np.logspace(0, 6, 10):
    estim_step, iterations = algorithm.compute_step(h0)
    print("h0 = %.3e, Approx. h* = %.3e (%d iterations)" % (h0, estim_step, iterations))

# %%
