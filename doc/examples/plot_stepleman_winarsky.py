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
# Use the method on a simple problem
# ----------------------------------

# %%
# In the next example, we use the algorithm on the exponential function.
# We create the :class:`~numericalderivative.SteplemanWinarsky` algorithm using the function and the point x.
# Then we use the :meth:`~numericalderivative.SteplemanWinarsky.compute_step()` method to compute the step,
# using an upper bound of the step as an initial point of the algorithm.
# Finally, use the :meth:`~numericalderivative.SteplemanWinarsky.compute_first_derivative()` method to compute
# an approximate value of the first derivative using finite differences.
# The :meth:`~numericalderivative.SteplemanWinarsky.get_number_of_function_evaluations()` method
# can be used to get the number of function evaluations.

# %%
x = 1.0
algorithm = nd.SteplemanWinarsky(np.exp, x, verbose=True)
initial_step = 1.0e0
step, number_of_iterations = algorithm.compute_step(initial_step)
f_prime_approx = algorithm.compute_first_derivative(step)
feval = algorithm.get_number_of_function_evaluations()
f_prime_exact = np.exp(x)  # Since the derivative of exp is exp.
print(f"Computed step = {step:.3e}")
print(f"Number of iterations = {number_of_iterations}")
print(f"f_prime_approx = {f_prime_approx}")
print(f"f_prime_exact = {f_prime_exact}")
absolute_error = abs(f_prime_approx - f_prime_exact)

# %%
# Use the method on the ScaledExponentialProblem
# ----------------------------------------------

# %%
# Consider this problem.

# %%
benchmark = nd.ScaledExponentialProblem()
name = benchmark.get_name()
x = benchmark.get_x()
third_derivative = benchmark.get_third_derivative()
third_derivative_value = third_derivative(x)
optimum_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
    third_derivative_value
)
print(f"Name = {name}, x = {x}")
print(f"Optimal step for central finite difference formula = {optimum_step}")
print(f"Minimum absolute error= {absolute_error}")


# %%
# Plot the error vs h
# -------------------

# %%
x = 1.0
finite_difference = nd.FirstDerivativeCentral(benchmark.function, x)
number_of_points = 1000
step_array = np.logspace(-7.0, 5.0, number_of_points)
error_array = np.zeros((number_of_points))
for i in range(number_of_points):
    h = step_array[i]
    f_prime_approx = finite_difference.compute(h)
    error_array[i] = abs(f_prime_approx - benchmark.first_derivative(x))

# %%
pl.figure()
pl.plot(step_array, error_array)
pl.plot([optimum_step] * 2, [min(error_array), max(error_array)], label=r"$h^*$")
pl.title("Central finite difference")
pl.xlabel("h")
pl.ylabel("Error")
pl.xscale("log")
pl.yscale("log")
pl.legend(bbox_to_anchor=(1, 1))
pl.tight_layout()


# %%
# Use the algorithm to detect h*

# %%
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
# Plot the absolute difference depending on the step
# --------------------------------------------------


# %%
def fd_difference(h1, h2, function, x):
    """
    Compute the difference of central difference approx. for different step sizes

    This function computes the absolute value of the difference of approximations
    evaluated at two different steps h1 and h2:

        d = abs(FD(h1) - FD(h2))

    where FD(h) is the approximation from the finite difference formula
    evaluated from the step h.

    Parameters
    ----------
    h1 : float, > 0
        The first step
    h2 : float, > 0
        The second step
    function : function
        The function
    x : float
        The input point where the derivative is approximated.
    """
    finite_difference = nd.FirstDerivativeCentral(function, x)
    f_prime_approx_1 = finite_difference.compute(h1)
    f_prime_approx_2 = finite_difference.compute(h2)
    diff_current = abs(f_prime_approx_1 - f_prime_approx_2)
    return diff_current


# %%
# Plot the evolution of | FD(h) - FD(h / 2) | for different values of h
number_of_points = 1000
step_array = np.logspace(-7.0, 5.0, number_of_points)
diff_array = np.zeros((number_of_points))
for i in range(number_of_points):
    h = step_array[i]
    diff_array[i] = fd_difference(h, h / 2, benchmark.function, x)

# %%
pl.figure()
pl.plot(step_array, diff_array)
pl.title("F.D. difference")
pl.xlabel("h")
pl.ylabel(r"$|\operatorname{FD}(h) - \operatorname{FD}(h / 2) |$")
pl.xscale("log")
pl.yscale("log")
pl.tight_layout()


# %%
# Plot the criterion depending on the step
# ----------------------------------------

# %%
# Plot the evolution of | FD(h) - FD(h / 2) | for different values of h
number_of_points = 20
h_initial = 1.0e5
beta = 4.0
step_array = np.zeros((number_of_points))
diff_array = np.zeros((number_of_points))
for i in range(number_of_points):
    if i == 0:
        step_array[i] = h_initial / beta
        diff_array[i] = fd_difference(step_array[i], h_initial, benchmark.function, x)
    else:
        step_array[i] = step_array[i - 1] / beta
        diff_array[i] = fd_difference(
            step_array[i], step_array[i - 1], benchmark.function, x
        )

# %%
pl.figure()
pl.plot(step_array, diff_array, "o")
pl.title("F.D. difference")
pl.xlabel("h")
pl.ylabel(r"$|\operatorname{FD}(h) - \operatorname{FD}(h / 2) |$")
pl.xscale("log")
pl.yscale("log")
pl.tight_layout()

# %%
# Compute reference step
# ----------------------

# %%
p = 1.0e-16
beta = 4.0
h_reference = beta * p ** (1 / 3) * x
print("Suggested h0 = ", h_reference)

# %%
# Plot number of lost digits vs h
# -------------------------------

# %%
# The :meth:`~numericalderivative.SteplemanWinarsky.number_of_lost_digits` method
# computes the number of lost digits in the approximated derivative
# depending on the step.

# %%
h = 1.0e4
print("Starting h = ", h)
n_digits = algorithm.number_of_lost_digits(h)
print("Number of lost digits = ", n_digits)
threshold = np.log10(p ** (-1.0 / 3.0) / beta)
print("Threshold = ", threshold)

step_zero, iterations = algorithm.search_step_with_bisection(
    1.0e-5,
    1.0e7,
)
print("step_zero = ", step_zero)
print("iterations = ", iterations)

estim_step, iterations = algorithm.compute_step(step_zero, beta=1.5)
print("estim_step = ", estim_step)
print("iterations = ", iterations)

# %%
number_of_points = 1000
step_array = np.logspace(-7.0, 7.0, number_of_points)
n_digits_array = np.zeros((number_of_points))
for i in range(number_of_points):
    h = step_array[i]
    n_digits_array[i] = algorithm.number_of_lost_digits(h)

# %%
y_max = algorithm.number_of_lost_digits(h_reference)
pl.figure()
pl.plot(step_array, n_digits_array, label="$N(h)$")
pl.plot([h_reference] * 2, [0.0, y_max], "--", label=r"$h_{ref}$")
pl.plot([step_zero] * 2, [0.0, y_max], "--", label=r"$h^{(0)}$")
pl.plot([estim_step] * 2, [0.0, y_max], "--", label=r"$h^\star$")
pl.plot(
    step_array,
    np.array([threshold] * number_of_points),
    ":",
    label=r"$T$",
)
pl.title("Number of digits lost by F.D.")
pl.xlabel("h")
pl.ylabel("$N(h)$")
pl.xscale("log")
_ = pl.legend(bbox_to_anchor=(1.0, 1.0))
pl.tight_layout()


# %%
pl.figure()
pl.plot(step_array, error_array)
pl.plot([step_zero] * 2, [0.0, 1.0e-9], "--", label=r"$h^{(0)}$")
pl.plot([estim_step] * 2, [0.0, 1.0e-9], "--", label=r"$h^\star$")
pl.title("Finite difference")
pl.xlabel("h")
pl.ylabel("Error")
pl.xscale("log")
pl.legend(bbox_to_anchor=(1.0, 1.0))
pl.yscale("log")
pl.tight_layout()

# %%
# Use the bisection search
# ------------------------


# %%
# In some cases, it is difficult to find the initial step.
# In this case, we can use the bisection algorithm, which can produce
# an initial guess for the step.c
# This algorithm is based on a search for a suitable step within
# an interval.

# %%
# Test with single point and default parameters.

# %%
x = 1.0
f_prime_approx, number_of_iterations = algorithm.search_step_with_bisection(
    1.0e-7,
    1.0e7,
)
feval = algorithm.get_number_of_function_evaluations()
print("FD(x) = ", f_prime_approx)
print("number_of_iterations = ", number_of_iterations)
print("Func. eval = ", feval)

# %%
# See how the algorithm behaves if we use or do not use the log scale
# when searching for the optimal step (this can be slower).

# %%
x = 1.0
maximum_bisection = 53
print("+ No log scale.")
h0, iterations = algorithm.search_step_with_bisection(
    1.0e-7,
    1.0e1,
    maximum_bisection=53,
    log_scale=False,
)
print("Pas initial = ", h0, ", iterations = ", iterations)
print("+ Log scale.")
h0, iterations = algorithm.search_step_with_bisection(
    1.0e-7,
    1.0e1,
    maximum_bisection=53,
    log_scale=True,
)
print("Pas initial = ", h0, ", iterations = ", iterations)

# %%
# In the next example, we search for an initial step using bisection,
# then use this step as an initial guess for the algorithm.
# Finally, we compute an approximation of the first derivative using
# the finite difference formula.

# %%
benchmark = nd.ExponentialProblem()
x = 1.0
algorithm = nd.SteplemanWinarsky(benchmark.function, x, verbose=True)
initial_step, estim_relative_error = algorithm.search_step_with_bisection(
    1.0e-6,
    100.0 * x,
    beta=4.0,
)
step, number_of_iterations = algorithm.compute_step(initial_step)
f_prime_approx = algorithm.compute_first_derivative(step)
absolute_error = abs(f_prime_approx - benchmark.first_derivative(x))
feval = algorithm.get_number_of_function_evaluations()
print(
    "x = %.3f, abs. error = %.3e, estim. rel. error = %.3e, Func. eval. = %d"
    % (x, absolute_error, estim_relative_error, number_of_function_evaluations)
)

#

# %%
