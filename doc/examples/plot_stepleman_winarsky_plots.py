#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Plot Stepleman & Winarsky's method
==================================

Find a step which is near to optimal for a central finite difference 
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
step_array = np.logspace(-15.0, 1.0, number_of_points)
n_digits_array = np.zeros((number_of_points))
algorithm = nd.SteplemanWinarsky(np.exp, x)
for i in range(number_of_points):
    h = step_array[i]
    n_digits_array[i] = algorithm.number_of_lost_digits(h)

pl.figure()
pl.plot(step_array, n_digits_array)
pl.title(r"Number of digits lost by F.D.. $f(x) = \exp(x)$")
pl.xlabel("h")
pl.ylabel("$N(h)$")
pl.xscale("log")

# %%
# Plot the number of lost digits for sin
x = 1.0
step_array = np.logspace(-7.0, 2.0, number_of_points)
n_digits_array = np.zeros((number_of_points))
algorithm = nd.SteplemanWinarsky(np.sin, x)
for i in range(number_of_points):
    h = step_array[i]
    n_digits_array[i] = algorithm.number_of_lost_digits(h)

# %%
pl.figure()
pl.plot(step_array, n_digits_array)
pl.title(r"Number of digits lost by F.D.. $f(x) = \sin(x)$")
pl.xlabel("h")
pl.ylabel("$N(h)$")
pl.xscale("log")

# %%
# For each function, at point x = 1, plot the error vs the step computed
# by the method


# %%
def plot_error_vs_h_with_SW_steps(
    name, function, function_derivative, x, step_array, h_min, h_max, verbose=False
):
    """
    Plot the computed error depending on the step for an array of F.D. steps

    Parameters
    ----------
    name : str
        The name of the problem
    function : function
        The function.
    first_derivative : function
        The exact first derivative of the function
    x : float
        The input point where the test is done
    step_array : list(float)
        The array of finite difference steps
    h_min : float, > 0
        The lower bound to bracket the initial differentiation step.
    h_max : float, > kmin
        The upper bound to bracket the initial differentiation step.
    verbose : bool, optional
        Set to True to print intermediate messages. The default is False.
    """
    algorithm = nd.SteplemanWinarsky(function, x)
    number_of_points = len(step_array)
    error_array = np.zeros((number_of_points))
    for i in range(number_of_points):
        h = step_array[i]
        f_prime_approx = algorithm.compute_first_derivative(h)
        error_array[i] = abs(f_prime_approx - function_derivative(x))

    bisection_h0_step, bisection_h0_iteration = algorithm.find_initial_step(
        h_min, h_max
    )
    step, bisection_iterations = algorithm.find_step(bisection_h0_step)

    if verbose:
        print(name)
        print(f"h_min = {h_min:.3e}, h_max = {h_max:.3e}")
        print(
            "Bisection initial_step = %.3e using %d iterations"
            % (bisection_h0_step, bisection_h0_iteration)
        )
        print("Bisection h* = %.3e using %d iterations" % (step, bisection_iterations))

    minimum_error = np.nanmin(error_array)
    maximum_error = np.nanmax(error_array)

    pl.figure()
    pl.plot(step_array, error_array)
    pl.plot(
        [h_min] * 2,
        [minimum_error, maximum_error],
        "--",
        label=r"$h_{\min}$",
    )
    pl.plot(
        [h_max] * 2,
        [minimum_error, maximum_error],
        "--",
        label=r"$h_{\max}$",
    )
    pl.plot(
        [bisection_h0_step] * 2,
        [minimum_error, maximum_error],
        "--",
        label="$h_{0}^{(B)}$",
    )
    pl.plot([step] * 2, [minimum_error, maximum_error], "--", label="$h^{*}$")
    pl.title("Finite difference : %s at point x = %.0f" % (name, x))
    pl.xlabel("h")
    pl.ylabel("Error")
    pl.xscale("log")
    pl.yscale("log")
    pl.legend(bbox_to_anchor=(1.0, 1.0))
    pl.subplots_adjust(right=0.8)
    return


# %%
def plot_error_vs_h_benchmark(problem, x, step_array, h_min, h_max, verbose=False):
    """
    Plot the computed error depending on the step for an array of F.D. steps

    Parameters
    ----------
    problem : nd.BenchmarkProblem
        The problem
    x : float
        The input point where the test is done
    step_array : list(float)
        The array of finite difference steps
    kmin : float, > 0
        The minimum step k for the second derivative.
    kmax : float, > kmin
        The maximum step k for the second derivative.
    verbose : bool, optional
        Set to True to print intermediate messages. The default is False.
    """
    plot_error_vs_h_with_SW_steps(
        problem.get_name(),
        problem.get_function(),
        problem.get_first_derivative(),
        x,
        step_array,
        h_min,
        h_max,
        True,
    )


# %%
problem = nd.ExponentialProblem()
x = 1.0
number_of_points = 1000
step_array = np.logspace(-15.0, 1.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, 1.0e-10, 1.0e0, True)

# %%
x = 12.0
step_array = np.logspace(-15.0, 1.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, 1.0e-10, 1.0e0)

if False:
    problem = nd.LogarithmicProblem()
    x = 1.0
    plot_error_vs_h_benchmark(problem, x, step_array, 1.0e-15, 1.0e0, True)

# %%
problem = nd.LogarithmicProblem()
x = 1.1
step_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, 1.0e-14, 1.0e-1, True)

# %%
problem = nd.SinProblem()
x = 1.0
step_array = np.logspace(-15.0, 0.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, 1.0e-15, 1.0e-0)

# %%
problem = nd.SquareRootProblem()
x = 1.0
step_array = np.logspace(-15.0, 0.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, 1.0e-15, 1.0e-0, True)

# %%
problem = nd.AtanProblem()
x = 1.0
step_array = np.logspace(-15.0, 0.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, 1.0e-15, 1.0e-0)

# %%
problem = nd.ExponentialProblem()
print("+ Sensitivity of SW step depending on initial_step")
print("Case 1 : exp")
x = 1.0
function = problem.get_function()
third_derivative = problem.get_third_derivative()
algorithm = nd.SteplemanWinarsky(
    function,
    x,
)
third_derivative_value = third_derivative(x)
optimal_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
    third_derivative_value
)
print("Exact h* = %.3e" % (optimal_step))
print("absolute_error = %.3e" % (absolute_error))
for initial_step in np.logspace(-4, 0, 10):
    estim_step, iterations = algorithm.find_step(initial_step)
    print("initial_step = %.3e, Approx. h* = %.3e (%d iterations)" % (initial_step, estim_step, iterations))

print("Case 2 : Scaled exp")
x = 1.0

# %%
problem = nd.ScaledExponentialProblem()
function = problem.get_function()
third_derivative = problem.get_third_derivative()
x = problem.get_x()
algorithm = nd.SteplemanWinarsky(function, x)
third_derivative_value = third_derivative(x)
optimal_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
    third_derivative_value
)
print("Exact h* = %.3e" % (optimal_step))
print("absolute_error = %.3e" % (absolute_error))
for initial_step in np.logspace(0, 6, 10):
    estim_step, iterations = algorithm.find_step(initial_step)
    print("initial_step = %.3e, Approx. h* = %.3e (%d iterations)" % (initial_step, estim_step, iterations))

# %%
