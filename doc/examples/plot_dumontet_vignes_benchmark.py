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
def benchmark_DumontetVignes_method(
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
        try:
            algorithm = nd.DumontetVignes(
                function, x, relative_precision=relative_precision, verbose=verbose
            )
            step, _ = algorithm.compute_step(kmin=kmin, kmax=kmax)
            f_prime_approx = algorithm.compute_first_derivative(step)
            number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
            exact_first_derivative = derivative_function(x)
            absolute_error = abs(f_prime_approx - exact_first_derivative)
            relative_error = absolute_error / abs(exact_first_derivative)
        except:
            number_of_function_evaluations = np.nan
            relative_error = np.nan
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
benchmark = nd.LogarithmicProblem()
f = benchmark.function
f_prime = benchmark.get_first_derivative()
kmin = 1.0e-9
kmax = 1.0e-3
relative_precision = 1.0e-14
absolute_error, feval = benchmark_DumontetVignes_method(
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
benchmark = nd.ExponentialProblem()
kmin = 1.0e-9
kmax = 1.0e0
relative_precision = 1.0e-14
average_relative_error, average_feval = benchmark_DumontetVignes_method(
    benchmark.function,
    benchmark.first_derivative,
    test_points,
    kmin,
    kmax,
    relative_precision,
    verbose=True,
)

# %%
# Map from the problem name to kmax

# %%
kmax_map = {
    "polynomial": 1.0,
    "inverse": 1.0e0,
    "exp": 1.0e-1,
    "log": 1.0e-3,  # x > 0
    "sqrt": 1.0e-3,  # x > 0
    "atan": 1.0e0,
    "sin": 1.0e0,
    "scaled exp": 1.0e5,
    "GMSW": 1.0e0,
    "SXXN1": 1.e0,
    "SXXN2": 1.e0,  # Fails
    "SXXN3": 1.e0,
    "SXXN4": 1.e0,
    "Oliver1": 1.e0,
    "Oliver2": 1.e0,
    "Oliver3": 1.e-3,
}



# %%
# Benchmark DumontetVignes
number_of_test_points = 100
relative_precision = 1.0e-14
data = []
function_list = nd.BuildBenchmark()
number_of_functions = len(function_list)
average_relative_error_list = []
average_feval_list = []
for i in range(number_of_functions):
    problem = function_list[i]
    name= problem.get_name()
    kmax = kmax_map[name]
    kmin = 1.e-16 * kmax
    function = problem.get_function()
    first_derivative = problem.get_first_derivative()
    interval = problem.get_interval()
    test_points = np.linspace(interval[0], interval[1], number_of_test_points)
    print(f"Function #{i}, {name}")
    average_relative_error, average_feval = benchmark_DumontetVignes_method(
        function,
        first_derivative,
        test_points,
        kmin,
        kmax,
        relative_precision,
        verbose=False,
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
        np.nanmean(average_relative_error_list),
        np.nanmean(average_feval_list),
    ]
)

tabulate.tabulate(
    data,
    headers=["Name", "kmin", "kmax", "Average rel. error", "Average func. eval"],
    tablefmt="html",
)
# %%
