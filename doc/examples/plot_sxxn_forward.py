#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Experiment with Shi, Xie, Xuan & Nocedal method
===============================================

Find a step which is near to optimal for a forward finite difference 
formula.

References
----------
- Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). 
  Adaptive finite-difference interval estimation for noisy derivative-free
  optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.
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
# We create the :class:`~numericalderivative.SXXNForward` algorithm using the function and the point x.
# Then we use the :meth:`~numericalderivative.SXXNForward.compute_step()` method to compute the step,
# using an upper bound of the step as an initial point of the algorithm.
# Finally, use the :meth:`~numericalderivative.SXXNForward.compute_first_derivative()` method to compute
# an approximate value of the first derivative using finite differences.
# The :meth:`~numericalderivative.SXXNForward.get_number_of_function_evaluations()` method
# can be used to get the number of function evaluations.

# %%
x = 1.0
algorithm = nd.SXXNForward(np.exp, x, verbose=True)
step, number_of_iterations = algorithm.compute_step()
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
problem = nd.ScaledExponentialProblem()
name = problem.get_name()
x = problem.get_x()
second_derivative = problem.get_second_derivative()
second_derivative_value = second_derivative(x)
optimum_step, absolute_error = nd.FirstDerivativeForward.compute_step(
    second_derivative_value
)
print(f"Name = {name}, x = {x}")
print(f"Optimal step for forward finite difference formula = {optimum_step}")
print(f"Minimum absolute error= {absolute_error}")


# %%
# Plot the error vs h
# -------------------

# %%
function = problem.get_function()
first_derivative = problem.get_first_derivative()
finite_difference = nd.FirstDerivativeForward(function, x)
number_of_points = 1000
step_array = np.logspace(-8.0, 4.0, number_of_points)
error_array = np.zeros((number_of_points))
for i in range(number_of_points):
    h = step_array[i]
    f_prime_approx = finite_difference.compute(h)
    error_array[i] = abs(f_prime_approx - first_derivative(x))

# %%
pl.figure()
pl.plot(step_array, error_array)
pl.plot([optimum_step] * 2, [min(error_array), max(error_array)], label=r"$h^*$")
pl.title("Forward finite difference")
pl.xlabel("h")
pl.ylabel("Error")
pl.xscale("log")
pl.yscale("log")
pl.legend(bbox_to_anchor=(1, 1))
pl.tight_layout()


# %%
# Use the algorithm to detect h*

# %%
algorithm = nd.SXXNForward(function, x, verbose=True)
x = 1.0e0
h_optimal, iterations = algorithm.compute_step()
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print("Optimum h =", h_optimal)
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
absolute_error = abs(f_prime_approx - problem.first_derivative(x))
print("Error = ", absolute_error)


# %%
# Plot the criterion depending on the step
# ----------------------------------------

# %%
# Plot the test ratio depending on h
problem = nd.SinProblem()
function = problem.get_function()
name = problem.get_name()
x = problem.get_x()
algorithm = nd.SXXNForward(function, x, verbose=True)
minimum_test_ratio, maximum_test_ratio = algorithm.get_ratio_min_max()
absolute_precision = 1.0e-15
number_of_points = 500
step_array = np.logspace(-10.0, 3.0, number_of_points)
test_ratio_array = np.zeros((number_of_points))
for i in range(number_of_points):
    test_ratio_array[i] = algorithm.compute_test_ratio(
        step_array[i], absolute_precision
    )

# %%
pl.figure()
pl.plot(step_array, test_ratio_array, "-", label="Test ratio")
pl.plot(step_array, [minimum_test_ratio] * number_of_points, "--", label="Min")
pl.plot(step_array, [maximum_test_ratio] * number_of_points, ":", label="Max")
pl.title(f"{name} at x = {x}. Test ratio.")
pl.xlabel("h")
pl.ylabel(r"$r$")
pl.xscale("log")
pl.yscale("log")
pl.legend()
pl.tight_layout()


# %%
# See the history of steps during the search
# ------------------------------------------

# %%
# In Shi, Xie, Xuan & Nocedal's method, the algorithm
# produces a sequence of steps :math:`(h_i)_{1 \leq i \leq n_{iter}}`
# where :math:`n_{iter} \in \mathbb{N}` is the number of iterations.
# These steps are meant to converge to an
# approximately optimal step of for the forward finite difference formula of the
# first derivative.
# The optimal step :math:`h^\star` for the central finite difference formula of the
# first derivative can be computed depending on the second derivative of the
# function.
# In the next example, we want to compute the absolute error between
# each intermediate step :math:`h_i` and the exact value :math:`h^\star`
# to see how close the algorithm gets to the exact step.
# The list of intermediate steps during the algorithm can be obtained
# thanks to the :meth:`~numericalderivative.SXXNForward.get_step_history` method.


# %%
# In the next example, we print the intermediate steps k during
# the bissection algorithm that searches for a step such
# that the L ratio is satisfactory.

# %%
problem = nd.ScaledExponentialProblem()
function = problem.get_function()
name = problem.get_name()
x = problem.get_x()
algorithm = nd.SXXNForward(function, x, verbose=True)
step, number_of_iterations = algorithm.compute_step()
step_h_history = algorithm.get_step_history()
print(f"Number of iterations = {number_of_iterations}")
print(f"History of steps h : {step_h_history}")
last_step_h = step_h_history[-1]
print(f"Last step h : {last_step_h}")

# %%
# Then we compute the exact step, using :meth:`~numericalderivative.FirstDerivativeForward.compute_step`.
second_derivative = problem.get_second_derivative()
second_derivative_value = second_derivative(x)
print(f"f^(2)(x) = {second_derivative_value}")
absolute_precision = 1.0e-16
exact_step, absolute_error = nd.FirstDerivativeForward.compute_step(
    second_derivative_value, absolute_precision
)
print(f"Optimal step k for f'(x) using forward F.D. = {exact_step}")

# %%
# Plot the absolute error between the exact step k and the intermediate k
# of the algorithm.
error_step_h = [abs(step_h_history[i] - exact_step) for i in range(len(step_h_history))]
fig = pl.figure(figsize=(4.0, 3.0))
pl.title(f"Shi, Xie, Xuan & Nocedal on {name} at x = {x}")
pl.plot(range(len(step_h_history)), error_step_h, "o-")
pl.xlabel("Iterations")
pl.ylabel(r"$|h_i - h^\star|$")
pl.yscale("log")
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
pl.tight_layout()

# %%
# The previous figure shows that the algorithm converges relatively fast.
# The absolute error does not evolve monotically.

# %%
