#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Experiments with Dumontet & Vignes method
=========================================

References
----------
- Dumontet, J., & Vignes, J. (1977). Détermination du pas optimal dans le calcul des dérivées sur ordinateur. RAIRO. Analyse numérique, 11 (1), 13-25.
"""

# %%
import numpy as np
import pylab as pl
import numericalderivative as nd
import sys
from matplotlib.ticker import MaxNLocator


# %%
# Use the method on a simple problem
# ----------------------------------

# %%
# In the next example, we use the algorithm on the exponential function.
# We create the :class:`~numericalderivative.DumontetVignes` algorithm using the function and the point x.
# Then we use the :meth:`~numericalderivative.DumontetVignes.find_step()` method to compute the step,
# using an upper bound of the step as an initial point of the algorithm.
# Finally, use the :meth:`~numericalderivative.DumontetVignes.compute_first_derivative()` method to compute
# an approximate value of the first derivative using finite differences.
# The :meth:`~numericalderivative.DumontetVignes.get_number_of_function_evaluations()` method
# can be used to get the number of function evaluations.

# %%
x = 1.0
kmin = 1.0e-10
kmax = 1.0e0
algorithm = nd.DumontetVignes(np.exp, x, verbose=True)
step, number_of_iterations = algorithm.find_step(kmin=kmin, kmax=kmax)
f_prime_approx = algorithm.compute_first_derivative(step)
feval = algorithm.get_number_of_function_evaluations()
f_prime_exact = np.exp(x)  # Since the derivative of exp is exp.
print(f"Computed step = {step:.3e}")
print(f"Number of iterations = {number_of_iterations}")
print(f"f_prime_approx = {f_prime_approx}")
print(f"f_prime_exact = {f_prime_exact}")
absolute_error = abs(f_prime_approx - f_prime_exact)

# %%
# Useful functions
# ----------------

# %%
# These functions will be used later in this example.


# %%
def compute_ell(function, x, k, relative_precision):
    """
    Compute the L ratio for a given value of the step k.
    """
    algorithm = nd.DumontetVignes(function, x, relative_precision=relative_precision)
    ell = algorithm.compute_ell(k)
    return ell


def compute_f3_inf_sup(function, x, k, relative_precision):
    """
    Compute the upper and lower bounds of the third derivative for a given value of k.
    """
    algorithm = nd.DumontetVignes(function, x, relative_precision=relative_precision)
    _, f3inf, f3sup = algorithm.compute_ell(k)
    return f3inf, f3sup


def plot_step_sensitivity(
    x,
    name,
    function,
    function_derivative,
    function_third_derivative,
    step_array,
    iteration_maximum=53,
    relative_precision=1.0e-15,
    kmin=None,
    kmax=None,
):
    """
    Create a plot representing the absolute error depending on step.

    Compute the approximate derivative using central F.D. formula.
    Plot the approximately optimal step computed by DumontetVignes.

    Parameters
    ----------
    x : float
        The input point
    name : str
        The name of the problem
    function : function
        The function.
    function_derivative : function
        The exact first derivative of the function.
    function_third_derivative : function
        The exact third derivative of the function.
    step_array : array(n_points)
        The array of steps to consider
    iteration_maximum : int
        The maximum number of iterations in DumontetVignes
    relative_precision : float, > 0
        The relative precision of the function evaluation
    kmin : float, kmin > 0
        A minimum bound for k. The default is None.
        If no value is provided, the default is to compute the smallest
        possible kmin using number_of_digits and x.
    kmax : float, kmax > kmin > 0
        A maximum bound for k. The default is None.
        If no value is provided, the default is to compute the largest
        possible kmax using number_of_digits and x.
    """
    print("+ ", name)
    # 1. Plot the error vs h
    algorithm = nd.DumontetVignes(function, x, verbose=True)
    number_of_points = len(step_array)
    error_array = np.zeros((number_of_points))
    for i in range(number_of_points):
        f_prime_approx = algorithm.compute_first_derivative(step_array[i])
        error_array[i] = abs(f_prime_approx - function_derivative(x))

    # 2. Algorithm to detect h*
    algorithm = nd.DumontetVignes(function, x, relative_precision=relative_precision)
    print("Exact f'''(x) = %.3e" % (function_third_derivative(x)))
    estim_step, _ = algorithm.find_step(
        iteration_maximum=iteration_maximum,
        kmin=kmin,
        kmax=kmax,
    )
    fprime = algorithm.compute_first_derivative(estim_step)
    number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
    print("Function evaluations =", number_of_function_evaluations)
    print("Estim. derivative = %.3e" % (fprime))
    print("Exact. derivative = %.3e" % (function_derivative(x)))
    f_prime_approx = algorithm.compute_first_derivative(estim_step)
    absolute_error = abs(f_prime_approx - function_derivative(x))
    print("Exact abs. error  = %.3e" % (absolute_error))
    print("Exact rel. error  = %.3e" % (absolute_error / abs(function_derivative(x))))
    # Compute exact step
    absolute_precision = abs(function(x) * relative_precision)
    third_derivative_value = function_third_derivative(x)
    optimal_step, optimal_error = nd.FirstDerivativeCentral.compute_step(
        third_derivative_value, absolute_precision
    )
    print("Exact step     = %.3e" % (optimal_step))
    print("Estimated step = %.3e" % (estim_step))
    print("Optimal abs. error = %.3e" % (optimal_error))
    print("Optimal rel. error = %.3e" % (optimal_error / abs(function_derivative(x))))

    minimum_error = np.nanmin(error_array)
    maximum_error = np.nanmax(error_array)

    pl.figure()
    pl.plot(step_array, error_array)
    pl.plot(
        [estim_step] * 2, [minimum_error, maximum_error], "--", label=r"$\widetilde{h}$"
    )
    pl.plot(optimal_step, optimal_error, "o", label=r"$h^\star$")
    pl.title("Finite difference for %s" % (name))
    pl.xlabel("h")
    pl.ylabel("Error")
    pl.xscale("log")
    pl.yscale("log")
    pl.legend(bbox_to_anchor=(1.1, 1.0))
    pl.tight_layout()
    return


# %%
def plot_ell_ratio(
    name,
    function,
    x,
    number_of_points,
    number_of_digits,
    relative_precision,
    kmin=None,
    kmax=None,
    y_logscale=False,
    plot_L_constants=True,
    epsilon_ell=1.0e-5,
):
    """Plot the ell ratio depending on the step size.

    This ell ratio is used in DumontetVignes."""
    ell_1 = 1.0 / 15.0  # Eq. 34, fixed
    ell_2 = 1.0 / 2.0
    ell_3 = 1.0 / ell_2
    ell_4 = 1.0 / ell_1

    if kmin is None:
        print("Set default kmin")
        kmin = x * 2 ** (-number_of_digits + 1)  # Eq. 26
    if kmax is None:
        print("Set default kmax")
        kmax = x * 2 ** (number_of_digits - 1)
    k_array = np.logspace(np.log10(kmin), np.log10(kmax), number_of_points)
    ell_array = np.zeros((number_of_points))
    for i in range(number_of_points):
        ell_array[i], _, _ = compute_ell(function, x, k_array[i], relative_precision)

    fig = pl.figure()
    pl.plot(k_array, ell_array, label="L")
    if plot_L_constants:
        indices = np.isfinite(ell_array)
        maximum_finite_ell = np.max(ell_array[indices])
        print(
            f"maximum_finite_ell = {maximum_finite_ell}, "
            f"maximum_finite_ell - 1 = {maximum_finite_ell - 1}"
        )

        if maximum_finite_ell <= 1.0 + epsilon_ell:
            print("maximum L is lower or equal to 1")
            pl.plot(
                k_array, [ell_1] * number_of_points, "--", label=f"$L_1$ = {ell_1:.3f}"
            )
            pl.plot(
                k_array, [ell_2] * number_of_points, ":", label=f"$L_2$ = {ell_2:.3f}"
            )
        else:
            print("maximum L is greater than 1")
            pl.plot(
                k_array, [ell_3] * number_of_points, ":", label=f"$L_3$ = {ell_3:.3f}"
            )
            pl.plot(
                k_array, [ell_4] * number_of_points, "--", label=f"$L_4$ = {ell_4:.3f}"
            )
        pl.legend(bbox_to_anchor=(1.0, 1.0))
    pl.title(f"{name}, x = {x:.2e}, p = {relative_precision:.2e}")
    pl.xlabel("k")
    pl.ylabel("L")
    pl.xscale("log")
    if y_logscale:
        pl.yscale("log")
    #
    pl.tight_layout()
    return


# %%
# Plot the L ratio for various problems
# -------------------------------------

# %%
# The L ratio is the criterion used in (Dumontet & Vignes, 1977) algorithm
# to find a satisfactory step k.
# The algorithm searches for a step k so that the L ratio is inside
# an interval.
# This is computed by the :meth:`~numericalderivative.DumontetVignes.compute_ell`
# method.
# In the next examples, we plot the L ratio depending on k
# for different functions.

# %%
# 1. Consider the :class:`~numericalderivative.ExponentialProblem` function.

problem = nd.ExponentialProblem()
x = problem.get_x()
problem

# %%
number_of_points = 200
relative_precision = 1.0e-15
number_of_digits = 53
plot_ell_ratio(
    problem.get_name(),
    problem.get_function(),
    x,
    number_of_points,
    number_of_digits,
    relative_precision,
    kmin=1.0e-7,
    kmax=1.0e-3,
    plot_L_constants=True,
)
_ = pl.ylim(-20.0, 20.0)

# %%
# See how the figure changes when the relative precision is
# increased: use 1.e-14 (instead of 1.e-15 in the previous example).

# %%
relative_precision = 1.0e-14
number_of_digits = 53
plot_ell_ratio(
    problem.get_name(),
    problem.get_function(),
    x,
    number_of_points,
    number_of_digits,
    relative_precision,
    kmin=1.0e-7,
    kmax=1.0e-3,
    plot_L_constants=True,
)
_ = pl.ylim(-20.0, 20.0)

# %%
# See what happens when the relative precision is reduced:
# here 1.e-16 instead of 1.e-14 in the previous example.

# %%
relative_precision = 1.0e-16
number_of_digits = 53
plot_ell_ratio(
    problem.get_name(),
    problem.get_function(),
    x,
    number_of_points,
    number_of_digits,
    relative_precision,
    kmin=1.0e-7,
    kmax=1.0e-3,
)
_ = pl.ylim(-20.0, 20.0)


# %%
# We see that it is difficult to find a value of k such that
# L(k) is in the required interval when the relative precision is too
# close to zero.

# %%
# Plot the error depending on the step
# ------------------------------------

# %%
# In the next examples, we plot the error of the approximation of the
# first derivative by the finite difference formula depending
# on the step size.

# %%
x = 4.0
problem = nd.ExponentialProblem()
function = problem.get_function()
absolute_precision = sys.float_info.epsilon * function(x)
print("absolute_precision = %.3e" % (absolute_precision))

# %%
x = 4.1  # A carefully chosen point
relative_precision = sys.float_info.epsilon
number_of_points = 200
step_array = np.logspace(-15.0, 0.0, number_of_points)
kmin = 1.0e-5
kmax = 1.0e-2
plot_step_sensitivity(
    x,
    problem.get_name(),
    problem.get_function(),
    problem.get_first_derivative(),
    problem.get_third_derivative(),
    step_array,
    iteration_maximum=20,
    relative_precision=1.0e-15,
    kmin=kmin,
    kmax=kmax,
)

# %%
# In the previous figure, we see that the error reaches
# a minimum, which is indicated by the green point labeled :math:`h^\star`.
# The vertical dotted line represents the approximately optimal step :math:`\widetilde{h}`
# returned by the :meth:`~numericalderivative.DumontetVignes.find_step` method.
# We see that the method correctly computes an approximation of the the optimal step.


# %%
# Consider the :class:`~numericalderivative.ScaledExponentialProblem`.
# First, we plot the L ratio.

relative_precision = 1.0e-14
problem = nd.ScaledExponentialProblem()
number_of_digits = 53
plot_ell_ratio(
    problem.get_name(),
    problem.get_function(),
    problem.get_x(),
    number_of_points,
    number_of_digits,
    relative_precision,
    kmin=1.0e-1,
    kmax=1.0e2,
    plot_L_constants=True,
)

# %%
# Then plot the error depending on the step size.

# %%
problem = nd.ScaledExponentialProblem()
step_array = np.logspace(-7.0, 6.0, number_of_points)
plot_step_sensitivity(
    problem.get_x(),
    problem.get_name(),
    problem.get_function(),
    problem.get_first_derivative(),
    problem.get_third_derivative(),
    step_array,
    relative_precision=1.0e-15,
    kmin=1.0e-2,
    kmax=1.0e2,
)

# %%
# The previous figure shows that the optimal step is close to
# :math:`10^1`, which may be larger than what we may typically expect
# as a step size for a finite difference formula.


# %%
# Compute the lower and upper bounds of the third derivative
# ----------------------------------------------------------

# %%
# The algorithm is based on bounds of the third derivative, which is
# computed by the :meth:`~numericalderivative.DumontetVignes.compute_ell` method.
# These bounds are used to find a step which is approximately
# optimal to compute the step of the finite difference formula
# used for the first derivative.
# Hence, it is interesting to compare the bounds computed
# by the (Dumontet & Vignes, 1977) algorithm and the
# actual value of the third derivative.
# To compute the true value of the third derivative,
# we use two different methods:
#
# - a finite difference formula, using :class:`~numericalderivative.ThirdDerivativeCentral`,
# - the exact third derivative, using :meth:`~numericalderivative.SquareRootProblem.get_third_derivative`.

# %%
x = 1.0
k = 1.0e-3  # A first guess
print("x = ", x)
print("k = ", k)
problem = nd.SquareRootProblem()
function = problem.get_function()
finite_difference = nd.ThirdDerivativeCentral(function, x)
approx_f3d = finite_difference.compute(k)
print("Approx. f''(x) = ", approx_f3d)
third_derivative = problem.get_third_derivative()
exact_f3d = third_derivative(x)
print("Exact f''(x) = ", exact_f3d)

# %%
relative_precision = 1.0e-14
print("relative_precision = ", relative_precision)
function = problem.get_function()
f3inf, f3sup = compute_f3_inf_sup(function, x, k, relative_precision)
print("f3inf = ", f3inf)
print("f3sup = ", f3sup)

# %%
# The previous outputs shows that the lower and upper bounds
# computed by the algorithm contain, indeed, the true value of the
# third derivative in this case.

# %%
# The algorithm is based on finding an approximatly optimal
# step k to compute the third derivative.
# The next script computes the error of the central formula for the
# finite difference formula depending on the step k.

# %%
number_of_points = 200
function = problem.get_function()
third_derivative = problem.get_third_derivative()
k_array = np.logspace(-6.0, -1.0, number_of_points)
error_array = np.zeros((number_of_points))
algorithm = nd.ThirdDerivativeCentral(function, x)
for i in range(number_of_points):
    f2nde_approx = algorithm.compute(k_array[i])
    error_array[i] = abs(f2nde_approx - third_derivative(x))

# %%
pl.figure()
pl.plot(k_array, error_array)
pl.title("F. D. of 3de derivative for %s" % (problem.get_name()))
pl.xlabel("k")
pl.ylabel("Error")
pl.xscale("log")
pl.yscale("log")
#
pl.tight_layout()

# %%
# Plot the lower and upper bounds of the third derivative
# -------------------------------------------------------

# %%
# The next figure presents the sensitivity of the
# lower and upper bounds of the third derivative to the step k.
# Moreover, it presents the approximation of the third derivative
# using the central finite difference formula.
# This makes it possible to check that the lower and upper bounds
# actually contain the approximation produced by the F.D. formula.

# %%
problem = nd.SquareRootProblem()
number_of_points = 200
relative_precision = 1.0e-16
k_array = np.logspace(-5.0, -4.0, number_of_points)
f3_array = np.zeros((number_of_points, 3))
function = problem.get_function()
algorithm = nd.ThirdDerivativeCentral(function, x)
for i in range(number_of_points):
    f3inf, f3sup = compute_f3_inf_sup(function, x, k_array[i], relative_precision)
    f3_approx = algorithm.compute(k_array[i])
    f3_array[i] = [f3inf, f3_approx, f3sup]

# %%
pl.figure()
pl.plot(k_array, f3_array[:, 0], ":", label="f3inf")
pl.plot(k_array, f3_array[:, 1], "-", label="$D^{(3)}_f$")
pl.plot(k_array, f3_array[:, 2], ":", label="f3sup")
pl.title(f"F.D. of 3de derivative for {problem.get_name()} at x = {x}")
pl.xlabel("k")
pl.xscale("log")
pl.legend(bbox_to_anchor=(1.0, 1.0))
pl.tight_layout(pad=1.2)


# %%
x = 1.0e-2
relative_precision = 1.0e-14
number_of_digits = 53
plot_ell_ratio(
    problem.get_name(),
    problem.get_function(),
    x,
    number_of_points,
    number_of_digits,
    relative_precision,
    kmin=4.4e-7,
    kmax=1.0e-5,
    plot_L_constants=True,
)
_ = pl.ylim(-20.0, 20.0)

# %%
# The next example searches the optimal step for the square root function.

# %%
x = 1.0e-2
relative_precision = 1.0e-14
kmin = 1.0e-8
kmax = 1.0e-3
verbose = True
function = problem.get_function()
algorithm = nd.DumontetVignes(
    function, x, relative_precision=relative_precision, verbose=verbose
)
h_optimal, _ = algorithm.find_step(kmax=kmax)
print("h optimal = %.3e" % (h_optimal))
number_of_feval = algorithm.get_number_of_function_evaluations()
print(f"number_of_feval = {number_of_feval}")
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
feval = algorithm.get_number_of_function_evaluations()
first_derivative = problem.get_first_derivative()
absolute_error = abs(f_prime_approx - first_derivative(x))
print("Abs. error = %.3e" % (absolute_error))

ell_kmin, f3inf, f3sup = algorithm.compute_ell(kmin)
print("L(kmin) = ", ell_kmin)
ell_kmax, f3inf, f3sup = algorithm.compute_ell(kmax)
print("L(kmax) = ", ell_kmax)


# %%
# Consider the :class:`~numericalderivative.SinProblem`.

# %%
x = 1.0
relative_precision = 1.0e-14
problem = nd.SinProblem()
function = problem.get_function()
name = "sin"
number_of_digits = 53
kmin = 1.0e-5
kmax = 1.0e-3
plot_ell_ratio(
    name,
    function,
    x,
    number_of_points,
    number_of_digits,
    relative_precision,
    kmin=kmin,
    kmax=kmax,
    plot_L_constants=True,
)


# %%
x = 1.0
k = 1.0e-3
print("x = ", x)
print("k = ", k)
function = problem.get_function()
algorithm = nd.ThirdDerivativeCentral(function, x)
approx_f3d = algorithm.compute(k)
print("Approx. f''(x) = ", approx_f3d)
third_derivative = problem.get_third_derivative()
exact_f3d = third_derivative(x)
print("Exact f''(x) = ", exact_f3d)
relative_precision = 1.0e-14
print("relative_precision = ", relative_precision)
function = problem.get_function()
f3inf, f3sup = compute_f3_inf_sup(function, x, k, relative_precision)
print("f3inf = ", f3inf)
print("f3sup = ", f3sup)

# %%
# See the history of steps during the bissection search
# -----------------------------------------------------

# %%
# In Dumontet & Vignes's method, the bisection algorithm
# produces a sequence of steps :math:`(k_i)_{1 \leq i \leq n_{iter}}`
# where :math:`n_{iter} \in \mathbb{N}` is the number of iterations.
# These steps are meant to converge to an
# approximately optimal step of for the central finite difference formula of the
# third derivative.
# The optimal step :math:`k^\star` for the central finite difference formula of the
# third derivative can be computed depending on the fifth derivative of the
# function.
# In the next example, we want to compute the absolute error between
# each intermediate step :math:`k_i` and the exact value :math:`k^\star`
# to see how close the algorithm gets to the exact step.
# The list of intermediate steps during the algorithm can be obtained
# thanks to the :meth:`~numericalderivative.DumontetVignes.get_step_history` method.


# %%
# In the next example, we print the intermediate steps k during
# the bissection algorithm that searches for a step such
# that the L ratio is satisfactory.

# %%
problem = nd.SinProblem()
function = problem.get_function()
name = problem.get_name()
x = problem.get_x()
algorithm = nd.DumontetVignes(function, x, verbose=True)
kmin = 1.0e-10
kmax = 1.0e0
step, number_of_iterations = algorithm.find_step(kmin=kmin, kmax=kmax)
step_k_history = algorithm.get_step_history()
print(f"Number of iterations = {number_of_iterations}")
print(f"History of steps k : {step_k_history}")
last_step_k = step_k_history[-1]
print(f"Last step k : {last_step_k}")

# %%
# Then we compute the exact step, using :meth:`~numericalderivative.ThirdDerivativeCentral.compute_step`.

# %%
fifth_derivative = problem.get_fifth_derivative()
fifth_derivative_value = fifth_derivative(x)
print(f"f^(5)(x) = {fifth_derivative_value}")
absolute_precision = 1.0e-16
exact_step_k, absolute_error = nd.ThirdDerivativeCentral.compute_step(
    fifth_derivative_value, absolute_precision
)
print(f"Optimal step k for f^(3)(x) = {exact_step_k}")

# %%
# Plot the absolute error between the exact step k and the intermediate k
# of the algorithm.

# %%
error_step_k = [
    abs(step_k_history[i] - exact_step_k) for i in range(len(step_k_history))
]
fig = pl.figure()
pl.title(f"Dumontet & Vignes on {name} at x = {x}")
pl.plot(range(len(step_k_history)), error_step_k, "o-")
pl.xlabel("Iterations")
pl.ylabel(r"$|k_i - k^\star|$")
pl.yscale("log")
ax = fig.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
pl.tight_layout()

# %%
# The previous figure shows that the algorithm gets closer to the optimal
# value of the step k in the early iterations.
# In the last iterations of the algorithm, the absolute error does not
# continue to decrease monotically and produces a final absolute
# error close to :math:`10^{-3}`.

# %%
