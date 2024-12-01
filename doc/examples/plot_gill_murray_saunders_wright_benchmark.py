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
import tabulate
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
    step, _ = algorithm.compute_step(kmin, kmax)
    f_prime_approx = algorithm.compute_first_derivative(step)
    feval = algorithm.get_number_of_function_evaluations()
    absolute_error = abs(f_prime_approx - f_prime(x))
    return absolute_error, feval


# %%
def benchmark_method(
    function, derivative_function, test_points, kmin, kmax, verbose=False
):
    """
    Apply Stepleman & Winarsky method to compute the approximate first
    derivative using finite difference formula.

    Parameters
    ----------
    f : function
        The function.
    derivative_function : function
        The exact first derivative of the function
    test_points : list(float)
        The list of x points where the benchmark must be performed.
    verbose : bool, optional
        Set to True to print intermediate messages. The default is False.

    Returns
    -------
    average_relative_error : float, > 0
        The average relative error between the approximate first derivative
        and the true first derivative.
    feval : int
        The number of function evaluations.

    """
    number_of_test_points = len(test_points)
    relative_error_array = np.zeros(number_of_test_points)
    feval_array = np.zeros(number_of_test_points)
    for i in range(number_of_test_points):
        x = test_points[i]
        if verbose:
            print(f"x = {x:.3f}")
        (
            absolute_error,
            number_of_function_evaluations,
        ) = compute_first_derivative_GMSW(
            function,
            x,
            derivative_function,
            kmin,
            kmax,
        )
        relative_error = absolute_error / abs(derivative_function(x))
        if verbose:
            print(
                "x = %.3f, abs. error = %.3e, rel. error = %.3e, Func. eval. = %d"
                % (x, absolute_error, relative_error, number_of_function_evaluations)
            )
        relative_error_array[i] = relative_error
        feval_array[i] = number_of_function_evaluations

    average_relative_error = np.mean(relative_error_array)
    average_feval = np.mean(feval_array)
    if verbose:
        print("Average error =", average_relative_error)
        print("Average number of function evaluations =", average_feval)
    return average_relative_error, average_feval


# %%
print("+ Benchmark on several points")
number_of_test_points = 100
test_points = np.linspace(0.01, 12.2, number_of_test_points)
kmin = 1.0e-16
kmax = 1.0e-1
benchmark = nd.ExponentialProblem()
average_relative_error, average_feval = benchmark_method(
    benchmark.function, benchmark.first_derivative, test_points, kmin, kmax, True
)


# %%
function_list = [
    [nd.ExponentialProblem(), 1.0e-16, 1.0e-1],
    [nd.LogarithmicProblem(), 1.0e-16, 1.0e-3],
    [nd.SquareRootProblem(), 1.0e-16, 1.0e-3],
    [nd.AtanProblem(), 1.0e-16, 1.0e0],
    [nd.SinProblem(), 1.0e-16, 1.0e0],
    [nd.ScaledExponentialProblem(), 1.0e-10, 1.0e5],
]

# %%
# Benchmark GillMurraySaundersWright
number_of_test_points = 100
test_points = np.linspace(0.01, 12.2, number_of_test_points)
data = []
number_of_functions = len(function_list)
average_relative_error_list = []
average_feval_list = []
for i in range(number_of_functions):
    benchmark, kmin, kmax = function_list[i]
    name = benchmark.name
    average_relative_error, average_feval = benchmark_method(
        benchmark.function,
        benchmark.first_derivative,
        test_points,
        kmin,
        kmax,
    )
    average_relative_error_list.append(average_relative_error)
    average_feval_list.append(average_feval)
    data.append(
        (
            name,
            kmin,
            kmax,
            average_relative_error,
            average_feval,
        )
    )
data.append(
    [
        "Average",
        "-",
        "-",
        np.mean(average_relative_error_list),
        np.mean(average_feval_list),
    ]
)
tabulate.tabulate(
    data,
    headers=["Name", "kmin", "kmax", "Average error", "Average func. eval"],
    tablefmt="html",
)

# %%
