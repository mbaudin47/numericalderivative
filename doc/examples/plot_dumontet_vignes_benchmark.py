#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Benchmark Dumontet & Vignes method
==================================

Find a step which is near to optimal for a centered finite difference 
formula.

References
----------
- Dumontet, J., & Vignes, J. (1977). Détermination du pas optimal dans le calcul des dérivées sur ordinateur. RAIRO. Analyse numérique, 11 (1), 13-25.
"""
# %%
import numpy as np
import tabulate
import numericalderivative as nd


# %%
def benchmark_method(
    function,
    derivative_function,
    test_points,
    kmin,
    kmax,
    relative_precision,
    verbose=False,
):
    """
    Compute the first derivative using Dumontet & Vignes's method.

    Parameters
    ----------
    f : function
        The function.
    derivative_function : function
        The exact first derivative of the function.
    test_points : list(float)
        The list of x points where the derivative is to be evaluated
    kmin : float, > 0
        The minimum finite difference step
    kmax : float, > 0
        The maximum finite difference step
    relative_precision : float, > 0
        The relative precision of the function value
    verbose : bool
        Set to True to print intermediate messages.

    Returns
    -------
    average_relative_error : float, > 0
        The average relative error between the approximate first derivative
        and the exact first derivative
    average_feval : float
        The average number of function evaluations
    """
    number_of_test_points = len(test_points)
    relative_error_array = np.zeros(number_of_test_points)
    feval_array = np.zeros(number_of_test_points)
    for i in range(number_of_test_points):
        x = test_points[i]
        algorithm = nd.DumontetVignes(
            function, x, relative_precision=relative_precision, verbose=verbose
        )
        step, _ = algorithm.compute_step(kmin=kmin, kmax=kmax)
        f_prime_approx = algorithm.compute_first_derivative(step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        exact_first_derivative = derivative_function(x)
        absolute_error = abs(f_prime_approx - exact_first_derivative)
        relative_error = absolute_error / abs(exact_first_derivative)
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
        print("Average rel. error =", average_relative_error)
        print("Average number of function evaluations =", average_feval)
    return average_relative_error, average_feval


# %%
x = 1.1
benchmark = nd.LogarithmicDerivativeBenchmark()
f = benchmark.function
f_prime = benchmark.first_derivative
kmin = 1.0e-9
kmax = 1.0e-3
relative_precision = 1.0e-14
absolute_error, feval = benchmark_method(
    f,
    f_prime,
    [x],
    kmin,
    kmax,
    relative_precision,
    verbose=True,
)
print(f"absolute_error = {absolute_error}")
print(f"feval = {feval}")

# %%
print("+ Benchmark on several points")
number_of_test_points = 100
test_points = np.linspace(0.01, 12.5, number_of_test_points)
benchmark = nd.ExponentialDerivativeBenchmark()
kmin = 1.0e-9
kmax = 1.0e0
relative_precision = 1.0e-14
average_relative_error, average_feval = benchmark_method(
    benchmark.function,
    benchmark.first_derivative,
    test_points,
    kmin,
    kmax,
    relative_precision,
    verbose=True,
)

# %%
# Define a collection of benchmark problems
function_list = [
    [nd.ExponentialDerivativeBenchmark(), [1.0e-10, 1.0e-1]],
    [nd.LogarithmicDerivativeBenchmark(), [1.0e-10, 1.0e-3]],
    [nd.SquareRootDerivativeBenchmark(), [1.0e-10, 1.0e-3]],
    [nd.AtanDerivativeBenchmark(), [1.0e-10, 1.0e0]],
    [nd.SinDerivativeBenchmark(), [1.0e-10, 1.0e0]],
    [nd.ScaledExponentialDerivativeBenchmark(), [1.0e-10, 1.0e5]],
]


# %%
def benchmark_problem(benchmark, bracket, relative_precision, verbose=False):
    function = benchmark.function
    derivative = benchmark.first_derivative
    kmin, kmax = bracket
    average_relative_error, average_feval = benchmark_method(
        function,
        derivative,
        test_points,
        kmin,
        kmax,
        relative_precision,
        verbose=verbose,
    )
    return average_relative_error, average_feval


# %%
# Benchmark DumontetVignes
number_of_test_points = 100
relative_precision = 1.0e-14
test_points = np.linspace(0.01, 12.5, number_of_test_points)
data = []
number_of_functions = len(function_list)
average_relative_error_list = []
average_feval_list = []
for i in range(number_of_functions):
    benchmark, bracket_k = function_list[i]
    name = benchmark.name
    function = benchmark.function
    derivative = benchmark.first_derivative
    kmin, kmax = bracket_k
    average_relative_error, average_feval = benchmark_problem(
        benchmark, bracket_k, relative_precision
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
    headers=["Name", "kmin", "kmax", "Average rel. error", "Average func. eval"],
    tablefmt="html",
)
# %%
