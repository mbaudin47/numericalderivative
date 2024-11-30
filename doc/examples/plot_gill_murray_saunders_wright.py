#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Experiment with Gill, Murray, Saunders and Wright method
========================================================

Find a step which is near to optimal for a centered finite difference 
formula.

References
----------
- Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). Computing forward-difference intervals for numerical optimization. SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
"""

# %%
import numpy as np
import pylab as pl
import numericalderivative as nd


# %%
def compute_first_derivative_GMSW(
    f,
    x,
    f_prime,
    kmin,
    kmax,
    verbose=False,
):
    """
    Compute the approximate derivative from finite differences

    Parameters
    ----------
    f : function
        The function.
    x : float
        The point where the derivative is to be evaluated
    f_prime : function
        The exact first derivative of the function.
    kmin : float, > 0
        The minimum step k for the second derivative.
    kmax : float, > kmin
        The maximum step k for the second derivative.
    verbose : bool, optional
        Set to True to print intermediate messages. The default is False.

    Returns
    -------
    absolute_error : float, > 0
        The absolute error between the approximate first derivative
        and the true first derivative.

    feval : int
        The number of function evaluations.
    """
    algorithm = nd.GillMurraySaundersWright(f, x, verbose=verbose)
    step, number_of_iterations = algorithm.compute_step(kmin, kmax)
    f_prime_approx = algorithm.compute_first_derivative(step)
    feval = algorithm.get_number_of_function_evaluations()
    f_prime_exact = algorithm.compute_first_derivative(step)
    if verbose:
        print(f"Computed step = {step:.3e}")
        print(f"Number of iterations = {number_of_iterations}")
        print(f"f_prime_approx = {f_prime_approx}")
        print(f"f_prime_exact = {f_prime_exact}")
    absolute_error = abs(f_prime_approx - f_prime_exact)
    return absolute_error, feval



# %%
print("+ Test on ExponentialDerivativeBenchmark")
kmin = 1.0e-15
kmax = 1.0e1
x = 1.0
benchmark = nd.ExponentialDerivativeBenchmark()
optimal_step_formula = nd.FiniteDifferenceOptimalStep()
second_derivative_value = benchmark.second_derivative(x)
optimal_step, absolute_error = (
    optimal_step_formula.compute_step_first_derivative_forward(second_derivative_value)
)
print("Exact h* = %.3e" % (optimal_step))
(
    absolute_error,
    number_of_function_evaluations,
) = compute_first_derivative_GMSW(
    benchmark.function,
    x,
    benchmark.first_derivative,
    kmin, 
    kmax,
    verbose=True,
)
print(
    "x = %.3f, error = %.3e, Func. eval. = %d"
    % (x, absolute_error, number_of_function_evaluations)
)

# %%
print("+ Test on ScaledExponentialDerivativeBenchmark")
kmin = 1.0e-9
kmax = 1.0e8
x = 1.0
benchmark = nd.ScaledExponentialDerivativeBenchmark()
optimal_step_formula = nd.FiniteDifferenceOptimalStep()
second_derivative_value = benchmark.second_derivative(x)
optimal_step, absolute_error = (
    optimal_step_formula.compute_step_first_derivative_forward(second_derivative_value)
)
print("Exact h* = %.3e" % (optimal_step))
(
    absolute_error,
    number_of_function_evaluations,
) = compute_first_derivative_GMSW(
    benchmark.function,
    x,
    benchmark.first_derivative,
    kmin, 
    kmax,
    verbose=True,
)
print(
    "x = %.3f, error = %.3e, Func. eval. = %d"
    % (x, absolute_error, number_of_function_evaluations)
)


# %%
def benchmark_method(
    function, derivative_function, test_points, kmin, kmax, verbose=False
):
    """
    Apply Gill, Murray, Saunders & Wright method to compute the approximate first
    derivative using finite difference formula.

    Parameters
    ----------
    f : function
        The function.
    derivative_function : function
        The exact first derivative of the function
    test_points : list(float)
        The list of x points where the benchmark must be performed.
    kmin : float, > 0
        The minimum step k for the second derivative.
    kmax : float, > kmin
        The maximum step k for the second derivative.
    verbose : bool, optional
        Set to True to print intermediate messages. The default is False.

    Returns
    -------
    absolute_error : float, > 0
        The absolute error between the approximate first derivative
        and the true first derivative.

    feval : int
        The number of function evaluations.

    """
    number_of_test_points = len(test_points)
    relative_error_array = np.zeros(number_of_test_points)
    feval_array = np.zeros(number_of_test_points)
    for i in range(number_of_test_points):
        x = test_points[i]
        (
            absolute_error,
            number_of_function_evaluations,
        ) = compute_first_derivative_GMSW(
            function,
            x,
            derivative_function,
            kmin, 
            kmax,
            verbose
        )
        relative_error = absolute_error / abs(derivative_function(x))
        if verbose:
            print(
                "x = %.3f, abs. error = %.3e, rel. error = %.3e, Func. eval. = %d"
                % (x, absolute_error, relative_error, number_of_function_evaluations)
            )
        relative_error_array[i] = relative_error
        feval_array[i] = number_of_function_evaluations

    average_error = np.mean(relative_error_array)
    average_feval = np.mean(feval_array)
    if verbose:
        print("Average error =", average_error)
        print("Average number of function evaluations =", average_feval)
    return average_error, average_feval


# %%
print("+ Benchmark on several points")
number_of_test_points = 100
test_points = np.linspace(0.01, 12.2, number_of_test_points)
kmin = 1.0e-13
kmax = 1.0e-1
benchmark = nd.ExponentialDerivativeBenchmark()
average_error, average_feval = benchmark_method(
    benchmark.function, benchmark.first_derivative, test_points, kmin, kmax, True
)



# For each function, at point x = 1, plot the error vs the step computed
# by the method


# %%
def plot_error_vs_h_with_GMSW_steps(
    name, function, function_derivative, x, h_array, kmin, kmax, verbose=False
):
    algorithm = nd.GillMurraySaundersWright(function, x)
    number_of_points = len(h_array)
    error_array = np.zeros((number_of_points))
    for i in range(number_of_points):
        h = h_array[i]
        f_prime_approx = algorithm.compute_first_derivative(h)
        error_array[i] = abs(f_prime_approx - function_derivative(x))

    step, number_of_iterations = algorithm.compute_step(kmin, kmax)

    if verbose:
        print(name)
        print(f"Step h* = {step:.3e} using {number_of_iterations} iterations")

    minimum_error = np.nanmin(error_array)
    maximum_error = np.nanmax(error_array)

    pl.figure(figsize=(3.0, 2.0))
    pl.plot(h_array, error_array)
    pl.plot(
        [step] * 2,
        [minimum_error, maximum_error],
        "--",
        label=r"$\hat{h}$",
    )
    pl.title(f"(GMS & W). {name} at point x = {x}")
    pl.xlabel("h")
    pl.ylabel("Error")
    pl.xscale("log")
    pl.yscale("log")
    pl.legend(bbox_to_anchor=(1.0, 1.0))
    return


# %%
def plot_error_vs_h_benchmark(benchmark, x, h_array, kmin, kmax, verbose=False):
    plot_error_vs_h_with_GMSW_steps(
        benchmark.name,
        benchmark.function,
        benchmark.first_derivative,
        x,
        h_array,
        kmin, 
        kmax,
        verbose,
    )


# %%
benchmark = nd.ExponentialDerivativeBenchmark()
x = 1.0
number_of_points = 1000
h_array = np.logspace(-15.0, -1.0, number_of_points)
kmin = 1.0e-15
kmax = 1.0e-1
plot_error_vs_h_benchmark(benchmark, x, h_array, kmin, kmax, True)

# %%
x = 12.0
h_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(benchmark, x, h_array, kmin, kmax)

# %%
benchmark = nd.ScaledExponentialDerivativeBenchmark()
x = 1.0
kmin = 1.0e-10
kmax = 1.0e8
h_array = np.logspace(-10.0, 8.0, number_of_points)
plot_error_vs_h_benchmark(benchmark, x, h_array, kmin, kmax)

# %%
benchmark = nd.LogarithmicDerivativeBenchmark()
x = 1.1
kmin = 1.0e-14
kmax = 1.0e-1
h_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(benchmark, x, h_array, kmin, kmax, True)

# %%
benchmark = nd.SinDerivativeBenchmark()
x = 1.0
kmin = 1.0e-15
kmax = 1.0e-1
h_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(benchmark, x, h_array, kmin, kmax)

# %%
benchmark = nd.SquareRootDerivativeBenchmark()
x = 1.0
kmin = 1.0e-15
kmax = 1.0e-1
h_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(benchmark, x, h_array, kmin, kmax, True)

# %%
benchmark = nd.AtanDerivativeBenchmark()
x = 1.1
kmin = 1.0e-15
kmax = 1.0e-1
h_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(benchmark, x, h_array, kmin, kmax)

# %%
