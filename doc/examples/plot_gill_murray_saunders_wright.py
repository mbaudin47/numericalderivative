#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Experiment with Gill, Murray, Saunders and Wright method
========================================================

Find a step which is near to optimal for a central finite difference 
formula.

References
----------
- Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). Computing forward-difference intervals for numerical optimization. SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
"""

# %%
import numpy as np
import pylab as pl
import numericalderivative as nd
from matplotlib.ticker import MaxNLocator

# %%
# Use the method on a simple problem
# ----------------------------------

# %%
# In the next example, we use the algorithm on the exponential function.
# We create the :class:`~numericalderivative.GillMurraySaundersWright` algorithm using the function and the point x.
# Then we use the :meth:`~numericalderivative.GillMurraySaundersWright.compute_step()` method to compute the step,
# using an upper bound of the step as an initial point of the algorithm.
# Finally, use the :meth:`~numericalderivative.GillMurraySaundersWright.compute_first_derivative()` method to compute
# an approximate value of the first derivative using finite differences.
# The :meth:`~numericalderivative.GillMurraySaundersWright.get_number_of_function_evaluations()` method
# can be used to get the number of function evaluations.

# %%
x = 1.0
algorithm = nd.GillMurraySaundersWright(np.exp, x, verbose=True)
kmin = 1.0e-10
kmax = 1.0e0
step, number_of_iterations = algorithm.compute_step(kmin, kmax)
f_prime_approx = algorithm.compute_first_derivative(step)
feval = algorithm.get_number_of_function_evaluations()
f_prime_exact = np.exp(x)  # Since the derivative of exp is exp.
print(f"Computed step = {step:.3e}")
print(f"Number of iterations = {number_of_iterations}")
print(f"f_prime_approx = {f_prime_approx}")
print(f"f_prime_exact = {f_prime_exact}")
absolute_error = abs(f_prime_approx - f_prime_exact)


# %%
# Test the method on the exponential problem
# ------------------------------------------


# %%
def compute_first_derivative_GMSW(
    function,
    x,
    first_derivative,
    kmin,
    kmax,
    verbose=False,
):
    """
    Compute the approximate derivative from finite differences

    Parameters
    ----------
    function : function
        The function.
    x : float
        The point where the derivative is to be evaluated
    first_derivative : function
        The exact first derivative of the function.
    kmin : float, > 0
        The minimum step k for the second derivative.
    kmax : float, > kmin
        The maximum step k for the second derivative.
    verbose : bool, optional
        Set to True to print intermediate messages. The default is False.

    Returns
    -------
    relative_error : float, > 0
        The relative error between the approximate first derivative
        and the true first derivative.

    feval : int
        The number of function evaluations.
    """
    algorithm = nd.GillMurraySaundersWright(function, x, verbose=verbose)
    step, number_of_iterations = algorithm.compute_step(kmin, kmax)
    f_prime_approx = algorithm.compute_first_derivative(step)
    feval = algorithm.get_number_of_function_evaluations()
    f_prime_exact = first_derivative(x)
    if verbose:
        print(f"Computed step = {step:.3e}")
        print(f"Number of iterations = {number_of_iterations}")
        print(f"f_prime_approx = {f_prime_approx}")
        print(f"f_prime_exact = {f_prime_exact}")
    absolute_error = abs(f_prime_approx - f_prime_exact)
    return absolute_error, feval


# %%
print("+ Test on ExponentialProblem")
kmin = 1.0e-15
kmax = 1.0e1
x = 1.0
problem = nd.ExponentialProblem()
second_derivative_value = problem.second_derivative(x)
optimal_step, absolute_error = nd.FirstDerivativeForward.compute_step(
    second_derivative_value
)
print("Exact h* = %.3e" % (optimal_step))
(
    absolute_error,
    number_of_function_evaluations,
) = compute_first_derivative_GMSW(
    problem.function,
    x,
    problem.first_derivative,
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
problem = nd.ScaledExponentialProblem()
second_derivative = problem.get_second_derivative()
second_derivative_value = second_derivative(x)
optimal_step, absolute_error = nd.FirstDerivativeForward.compute_step(
    second_derivative_value
)
print("Exact h* = %.3e" % (optimal_step))
(
    absolute_error,
    number_of_function_evaluations,
) = compute_first_derivative_GMSW(
    problem.get_function(),
    x,
    problem.get_first_derivative(),
    kmin,
    kmax,
    verbose=True,
)
print(
    "x = %.3f, error = %.3e, Func. eval. = %d"
    % (x, absolute_error, number_of_function_evaluations)
)

# %%
# Benchmark the method on a collection of test points
# ---------------------------------------------------


# %%
def benchmark_method(
    function, first_derivative, test_points, kmin, kmax, verbose=False
):
    """
    Apply Gill, Murray, Saunders & Wright method to compute the approximate first
    derivative using finite difference formula.

    Parameters
    ----------
    f : function
        The function.
    first_derivative : function
        The exact first derivative of the function
    test_points : list(float)
        The list of x points where the problem must be performed.
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
        if verbose:
            print(f"i = {i}, x = {x:.3f}")
        (
            absolute_error,
            number_of_function_evaluations,
        ) = compute_first_derivative_GMSW(
            function, x, first_derivative, kmin, kmax, verbose
        )
        relative_error = absolute_error / abs(first_derivative(x))
        if verbose:
            print(
                f"x = {x:.3f}, abs. error = {absolute_error:.3e}, "
                f"rel. error = {relative_error:.3e}, "
                f"Func. eval. = {number_of_function_evaluations}"
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
print("+ Benchmark on several points on ScaledExponentialProblem")
number_of_test_points = 100
problem = nd.ScaledExponentialProblem()
interval = problem.get_interval()
test_points = np.linspace(interval[0], interval[1], number_of_test_points)
kmin = 1.0e-12
kmax = 1.0e1
average_error, average_feval = benchmark_method(
    problem.get_function(),
    problem.get_fifth_derivative(),
    test_points,
    kmin,
    kmax,
    True,
)

# %%
# Plot the condition error depending on the step
# ----------------------------------------------


# %%
def plot_condition_error(name, function, x, kmin, kmax, number_of_points=1000):
    # Plot the condition error depending on k.
    k_array = np.logspace(np.log10(kmin), np.log10(kmax), number_of_points)
    algorithm = nd.GillMurraySaundersWright(function, x)
    c_min, c_max = algorithm.get_threshold_min_max()
    condition_array = np.zeros((number_of_points))
    for i in range(number_of_points):
        condition_array[i] = algorithm.compute_condition(k_array[i])

    #
    pl.figure(figsize=(4.0, 3.0))
    pl.title(f"Condition error of {name} at x = {x}")
    pl.plot(k_array, condition_array)
    pl.plot([kmin, kmax], [c_min] * 2, label=r"$c_{min}$")
    pl.plot([kmin, kmax], [c_max] * 2, label=r"$c_{max}$")
    pl.xlabel(r"$h_\Phi$")
    pl.ylabel(r"$c(h_\Phi$)")
    pl.xscale("log")
    pl.yscale("log")
    pl.legend(bbox_to_anchor=(1.0, 1.0))
    pl.tight_layout()
    return


# %%
# The next plot presents the condition error :math:`c(h_\Phi)` depending
# on :math:`h_\Phi`.
# The two horizontal lines represent the minimum and maximum threshold
# values.
# We search for the value of :math:`h_\Phi` such that the condition
# error is between these two limits.

# %%
number_of_points = 1000
problem = nd.ScaledExponentialProblem()
x = problem.get_x()
name = problem.get_name()
function = problem.get_function()
kmin = 1.0e-10
kmax = 1.0e5
plot_condition_error(name, function, x, kmin, kmax)

# %%
# The previous plot shows that the condition error is a decreasing function
# of :math:`h_\Phi`.

# %%
# Remove the end points :math:`x = \pm \pi`, because sin has a zero
# second derivative at these points.
# This makes the algorithm fail.

# %%
print("+ Benchmark on several points on SinProblem")
number_of_test_points = 100
problem = nd.SinProblem()
interval = problem.get_interval()
epsilon = 1.0e-3
test_points = np.linspace(
    interval[0] + epsilon, interval[1] - epsilon, number_of_test_points
)
kmin = 1.0e-12
kmax = 1.0e1
average_error, average_feval = benchmark_method(
    problem.get_function(),
    problem.get_fifth_derivative(),
    test_points,
    kmin,
    kmax,
    True,
)


# %%
# Plot the condition error depending on k.
number_of_points = 1000
problem = nd.SinProblem()
x = -np.pi
name = problem.get_name()
function = problem.get_function()
kmin = 1.0e-20
kmax = 1.0e-10
plot_condition_error(name, function, x, kmin, kmax)

# %%
# In the previous plot, we see that there is no satisfactory
# value of :math:`h_\Phi` for the sin function
# at point :math:`x = -\pi`.

# %%
# Plot the error depending on the step
# ------------------------------------

# %%
# For each function, at point x = 1, plot the error vs the step computed
# by the method


# %%
def plot_error_vs_h_with_GMSW_steps(
    name, function, first_derivative, x, step_array, kmin, kmax, verbose=False
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
    kmin : float, > 0
        The minimum step k for the second derivative.
    kmax : float, > kmin
        The maximum step k for the second derivative.
    verbose : bool, optional
        Set to True to print intermediate messages. The default is False.
    """
    algorithm = nd.GillMurraySaundersWright(function, x)
    number_of_points = len(step_array)
    error_array = np.zeros((number_of_points))
    for i in range(number_of_points):
        f_prime_approx = algorithm.compute_first_derivative(step_array[i])
        error_array[i] = abs(f_prime_approx - first_derivative(x))

    step, number_of_iterations = algorithm.compute_step(kmin, kmax)

    if verbose:
        print(name)
        print(f"Step h* = {step:.3e} using {number_of_iterations} iterations")

    minimum_error = np.nanmin(error_array)
    maximum_error = np.nanmax(error_array)

    pl.figure()
    pl.plot(step_array, error_array)
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
    pl.tight_layout()
    return


# %%
def plot_error_vs_h_benchmark(problem, x, step_array, kmin, kmax, verbose=False):
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
    plot_error_vs_h_with_GMSW_steps(
        problem.get_name(),
        problem.get_function(),
        problem.get_first_derivative(),
        x,
        step_array,
        kmin,
        kmax,
        verbose,
    )


# %%
problem = nd.ExponentialProblem()
x = 1.0
number_of_points = 1000
step_array = np.logspace(-15.0, -1.0, number_of_points)
kmin = 1.0e-15
kmax = 1.0e-1
plot_error_vs_h_benchmark(problem, x, step_array, kmin, kmax, True)

# %%
x = 12.0
step_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, kmin, kmax)

# %%
problem = nd.ScaledExponentialProblem()
x = 1.0
kmin = 1.0e-10
kmax = 1.0e8
step_array = np.logspace(-10.0, 8.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, kmin, kmax)

# %%
problem = nd.LogarithmicProblem()
x = 1.1
kmin = 1.0e-14
kmax = 1.0e-1
step_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, kmin, kmax, True)

# %%
problem = nd.SinProblem()
x = 1.0
kmin = 1.0e-15
kmax = 1.0e-1
step_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, kmin, kmax)

# %%
problem = nd.SquareRootProblem()
x = 1.0
kmin = 1.0e-15
kmax = 1.0e-1
step_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, kmin, kmax, True)

# %%
problem = nd.AtanProblem()
x = 1.1
kmin = 1.0e-15
kmax = 1.0e-1
step_array = np.logspace(-15.0, -1.0, number_of_points)
plot_error_vs_h_benchmark(problem, x, step_array, kmin, kmax)

# %%
# See the history of steps during the bissection search
# -----------------------------------------------------

# %%
# In G, M, S & W's method, the bisection algorithm
# produces a sequence of steps :math:`(k_i)_{1 \leq i \leq n_{iter}}`
# where :math:`n_{iter} \in \mathbb{N}` is the number of iterations.
# These steps are meant to converge to an
# approximately optimal step of for the central finite difference formula of the
# second derivative.
# The optimal step :math:`k^\star` for the central finite difference formula of the
# second derivative can be computed depending on the fourth derivative of the
# function.
# In the next example, we want to compute the absolute error between
# each intermediate step :math:`k_i` and the exact value :math:`k^\star`
# to see how close the algorithm gets to the exact step.
# The list of intermediate steps during the algorithm can be obtained
# thanks to the :meth:`~numericalderivative.GillMurraySaundersWright.get_step_history` method.


# %%
# In the next example, we print the intermediate steps k during
# the bissection algorithm that searches for a step such
# that the L ratio is satisfactory.
# The algorithm has two different methods to update the step:
#
# - using the mean,
# - using the mean in the logarithm space (this is generally much faster).


def plot_GMSW_step_history(problem, kmin, kmax, logscale):
    function = problem.get_function()
    name = problem.get_name()
    x = problem.get_x()
    algorithm = nd.GillMurraySaundersWright(function, x, verbose=True)
    step, number_of_iterations = algorithm.compute_step(
        kmin=kmin, kmax=kmax, logscale=logscale
    )
    step_k_history = algorithm.get_step_history()
    print(f"Number of iterations = {number_of_iterations}")
    print(f"History of steps k : {step_k_history}")
    last_step_k = step_k_history[-1]
    print(f"Last step k : {last_step_k}")

    # Then we compute the exact step, using :meth:`~numericalderivative.SecondDerivativeCentral.compute_step`.
    fourth_derivative = problem.get_fourth_derivative()
    fourth_derivative_value = fourth_derivative(x)
    print(f"f^(4)(x) = {fourth_derivative_value}")
    absolute_precision = 1.0e-16
    exact_step_k, absolute_error = nd.SecondDerivativeCentral.compute_step(
        fourth_derivative_value, absolute_precision
    )
    print(f"Optimal step k for f^(2)(x) = {exact_step_k}")

    # Plot the absolute error between the exact step k and the intermediate k
    # of the algorithm.
    error_step_k = [
        abs(step_k_history[i] - exact_step_k) for i in range(len(step_k_history))
    ]
    fig = pl.figure()
    pl.title(f"GMSW on {name} at x = {x}. Log scale = {logscale}")
    pl.plot(range(len(step_k_history)), error_step_k, "o-")
    pl.xlabel("Iterations")
    pl.ylabel(r"$|k_i - k^\star|$")
    pl.yscale("log")
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    pl.tight_layout()


# %%
# First, test the logarithmic log scale.
problem = nd.SinProblem()
kmin = 1.0e-15
kmax = 1.0e3
logscale = True
plot_GMSW_step_history(problem, kmin, kmax, logscale)

# %%
# The previous figure shows that the algorithm does not necessarily
# reduce the distance to the optimal step when we use the logarithmic scale.
# The algorithm quickly stops and gets an error approximately equal to :math:`10^{-4}`.

# %%
# Secondly, test the ordinary scale, using the mean.
problem = nd.SinProblem()
kmin = 1.0e-15
kmax = 1.0e3
logscale = False
plot_GMSW_step_history(problem, kmin, kmax, logscale)

# %%
# In the previous plot, we see that the error first decreases
# down to an error approximately equal to :math:`10^{-4}`.
# Then the error slightly increases before the algorithm stops.

# %%
