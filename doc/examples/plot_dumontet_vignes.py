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


# %%
def compute_ell(function, x, k, relative_precision):
    algorithm = nd.DumontetVignes(function, x, relative_precision=relative_precision)
    ell = algorithm.compute_ell(k)
    return ell


def compute_f3_inf_sup(function, x, k, relative_precision):
    algorithm = nd.DumontetVignes(function, x, relative_precision=relative_precision)
    ell, f3inf, f3sup = algorithm.compute_ell(k)
    return f3inf, f3sup


def perform(
    x,
    name,
    function,
    function_derivative,
    function_third_derivative,
    h_array,
    iteration_maximum=53,
    relative_precision=1.0e-15,
    kmin=None,
    kmax=None,
):
    print("+ ", name)
    # 1. Plot the error vs h
    algorithm = nd.DumontetVignes(function, x, verbose=True)
    error_array = np.zeros((number_of_points))
    for i in range(number_of_points):
        f_prime_approx = algorithm.compute_first_derivative(h_array[i])
        error_array[i] = abs(f_prime_approx - function_derivative(x))

    # 2. Algorithm to detect h*
    algorithm = nd.DumontetVignes(function, x, relative_precision=relative_precision)
    print("Exact f'''(x) = %.3e" % (function_third_derivative(x)))
    estim_step, number_of_iterations = algorithm.compute_step(
        iteration_maximum=iteration_maximum,
        markdown=False,
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
    fdstep = nd.FiniteDifferenceOptimalStep(absolute_precision)
    third_derivative_value = function_third_derivative(x)
    optimal_step, optimal_error = fdstep.compute_step_first_derivative_central(
        third_derivative_value
    )
    print("Exact step     = %.3e" % (optimal_step))
    print("Estimated step = %.3e" % (estim_step))
    print("Optimal abs. error = %.3e" % (optimal_error))
    print("Optimal rel. error = %.3e" % (optimal_error / abs(function_derivative(x))))

    minimum_error = np.nanmin(error_array)
    maximum_error = np.nanmax(error_array)

    pl.figure(figsize=(3.0, 2.0))
    pl.plot(h_array, error_array)
    pl.plot(
        [estim_step] * 2, [minimum_error, maximum_error], "--", label=r"$\tilde{h}$"
    )
    pl.plot(optimal_step, optimal_error, "o", label=r"$h^\star$")
    pl.title("Finite difference for %s" % (name))
    pl.xlabel("h")
    pl.ylabel("Error")
    pl.xscale("log")
    pl.legend(bbox_to_anchor=(1.1, 1.0))
    pl.yscale("log")
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
    plot_L_constants=False,
):
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
        ell_array[i], f3inf, f3sup = compute_ell(function, x, k_array[i], relative_precision)

    pl.figure(figsize=(4.0, 3.0))
    pl.plot(k_array, ell_array)
    if plot_L_constants:
        indices = np.isfinite(ell_array)
        maximum_finite_ell = np.max(ell_array[indices])
        print("Maximum L = %.3e" % (maximum_finite_ell))
        if maximum_finite_ell < 1.0:
            pl.plot(k_array, [ell_1] * number_of_points, "--", label="$L_1$")
            pl.plot(k_array, [ell_2] * number_of_points, ":", label="$L_2$")
        else:
            pl.plot(k_array, [ell_3] * number_of_points, ":", label="$L_3$")
            pl.plot(k_array, [ell_4] * number_of_points, "--", label="$L_4$")
        pl.legend()
    pl.title("%s, x = %.3e, p = %.3e" % (name, x, relative_precision))
    pl.xlabel("k")
    pl.ylabel("L")
    pl.xscale("log")
    if y_logscale:
        pl.yscale("log")
    return


# %%
# 1. Exponential

number_of_points = 1000
relative_precision = 1.0e-15
x = 1.0
function = nd.ExponentialProblem().get_function()
name = "exp"
number_of_digits = 53
kmin = 1.55e-5
kmax = 1.0e-4
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
relative_precision = 1.0e-10
x = 1.0
function = nd.ExponentialProblem().get_function()
name = "exp"
number_of_digits = 53
kmin = 5.0e-5
kmax = 1.0e-2
plot_ell_ratio(
    name,
    function,
    x,
    number_of_points,
    number_of_digits,
    relative_precision,
    kmin=kmin,
    kmax=kmax,
    y_logscale=False,
    plot_L_constants=True,
)
pl.ylim(-20.0, 20.0)

# %%
relative_precision = 1.0e-14
x = 1.0
function = nd.ExponentialProblem().get_function()
name = "exp"
number_of_digits = 53
kmin = 4.0e-5
kmax = 1.0e-2
plot_ell_ratio(
    name,
    function,
    x,
    number_of_points,
    number_of_digits,
    relative_precision,
    kmin=kmin,
    kmax=kmax,
)

# %%
relative_precision = 1.0e-16
x = 4.0
function = nd.ExponentialProblem().get_function()
name = "exp"
number_of_digits = 53
kmin = 1.0e-5
kmax = 1.0e-2
plot_ell_ratio(
    name,
    function,
    x,
    number_of_points,
    number_of_digits,
    relative_precision,
    kmin=kmin,
    kmax=kmax,
)

# %%
relative_precision = 1.0e-14
x = 4.0
function = nd.ExponentialProblem().get_function()
name = "exp"
number_of_digits = 53
kmin = 3.2e-5
kmax = 1.0e-2
plot_ell_ratio(
    name,
    function,
    x,
    number_of_points,
    number_of_digits,
    relative_precision,
    kmin=kmin,
    kmax=kmax,
    y_logscale=False,
    plot_L_constants=True,
)
pl.ylim(-20.0, 20.0)

# %%
x = 4.0
benchmark = nd.ExponentialProblem()
function = benchmark.get_function()
absolute_precision = sys.float_info.epsilon * function(x)
print("absolute_precision = %.3e" % (absolute_precision))

# %%
x = 4.1
function = benchmark.get_function()
relative_precision = sys.float_info.epsilon
name = "exp"
function_derivative = benchmark.get_first_derivative()
function_third_derivative = benchmark.get_third_derivative()
number_of_points = 1000
h_array = np.logspace(-15.0, 0.0, number_of_points)
kmin = 1.0e-5
kmax = 1.0e-2
perform(
    x,
    name,
    function,
    function_derivative,
    function_third_derivative,
    h_array,
    iteration_maximum=20,
    relative_precision=1.0e-15,
    kmin=kmin,
    kmax=kmax,
)

# %%
# 2. Scaled exponential

x = 1.0
relative_precision = 1.0e-14
function = nd.ScaledExponentialProblem().get_function()
name = "scaled exp"
number_of_digits = 53
kmin = 1.0e-1
kmax = 1.0e2
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
name = "scaled exp"
benchmark = nd.ScaledExponentialProblem()
function = benchmark.function
function_derivative = benchmark.first_derivative
function_third_derivative = benchmark.third_derivative
number_of_points = 1000
h_array = np.logspace(-7.0, 6.0, number_of_points)
kmin = 1.0e-2
kmax = 1.0e2
relative_precision = 1.0e-15
perform(
    x,
    name,
    function,
    function_derivative,
    function_third_derivative,
    h_array,
    relative_precision=relative_precision,
    kmin=kmin,
    kmax=kmax,
)

# %%
#
print("+ 3. Square root")

x = 1.0
relative_precision = 1.0e-14
function = nd.SquareRootProblem().get_function()
name = "square root"
number_of_digits = 53
kmin = 4.3e-5
kmax = 1.0e-4
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
pl.ylim(-20.0, 20.0)


# %%
x = 1.0
k = 1.0e-3
print("x = ", x)
print("k = ", k)
benchmark = nd.SquareRootProblem()
finite_difference = nd.FiniteDifferenceFormula(benchmark.function, x)
approx_f3d = finite_difference.compute_third_derivative(k)
print("Approx. f''(x) = ", approx_f3d)
exact_f3d = benchmark.third_derivative(x)
print("Exact f''(x) = ", exact_f3d)

# %%
relative_precision = 1.0e-14
print("relative_precision = ", relative_precision)
f3inf, f3sup = compute_f3_inf_sup(benchmark.function, x, k, relative_precision)
print("f3inf = ", f3inf)
print("f3sup = ", f3sup)

# %%
number_of_points = 1000
k_array = np.logspace(-6.0, -1.0, number_of_points)
error_array = np.zeros((number_of_points))
for i in range(number_of_points):
    algorithm = nd.FiniteDifferenceFormula(benchmark.function, x)
    f2nde_approx = algorithm.compute_third_derivative(k_array[i])
    error_array[i] = abs(f2nde_approx - benchmark.third_derivative(x))

# %%
pl.figure(figsize=(3.0, 2.0))
pl.plot(k_array, error_array)
pl.title("Finite difference of 3de derivative for %s" % (name))
pl.xlabel("k")
pl.ylabel("Error")
pl.xscale("log")
pl.yscale("log")


# %%
number_of_points = 1000
relative_precision = 1.0e-16
k_array = np.logspace(-4.9, -4.0, number_of_points)
f3_array = np.zeros((number_of_points, 3))
function = benchmark.get_function()
for i in range(number_of_points):
    f3inf, f3sup = compute_f3_inf_sup(function, x, k_array[i], relative_precision)
    algorithm = nd.FiniteDifferenceFormula(function, x)
    f3_approx = algorithm.compute_third_derivative(k_array[i])
    f3_array[i] = [f3inf, f3_approx, f3sup]

pl.figure(figsize=(3.0, 2.0))
pl.plot(k_array, f3_array[:, 0], ":", label="f3inf")
pl.plot(k_array, f3_array[:, 1], "-", label="$D^{(3)}_f$")
pl.plot(k_array, f3_array[:, 2], ":", label="f3sup")
pl.title("Finite difference of 3de derivative for %s" % (name))
pl.xlabel("k")
pl.xscale("log")
pl.yscale("log")
pl.legend()


# %%
x = 1.0e-2
relative_precision = 1.0e-14
function = benchmark.get_function()
name = "square root"
number_of_digits = 53
kmin = 4.4e-7
kmax = 1.0e-4
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
pl.ylim(-20.0, 20.0)

# %%
# Search step
x = 1.0e-2
relative_precision = 1.0e-14
kmin = 1.0e-8
kmax = 1.0e-3
verbose = True
function = benchmark.get_function()
algorithm = nd.DumontetVignes(
    function, x, relative_precision=relative_precision, verbose=verbose
)
h_optimal, _ = algorithm.compute_step(kmax=kmax)
print("h optimal = %.3e" % (h_optimal))
number_of_feval = algorithm.get_number_of_function_evaluations()
print(f"number_of_feval = {number_of_feval}")
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
feval = algorithm.get_number_of_function_evaluations()
first_derivative = benchmark.get_first_derivative()
absolute_error = abs(f_prime_approx - first_derivative(x))
print("Abs. error = %.3e" % (absolute_error))

ell_kmin, f3inf, f3sup = algorithm.compute_ell(kmin)
print("L(kmin) = ", ell_kmin)
ell_kmax, f3inf, f3sup = algorithm.compute_ell(kmax)
print("L(kmax) = ", ell_kmax)


# %%
#
print("+ 4. sin")
x = 1.0
relative_precision = 1.0e-14
benchmark = nd.SinProblem()
function = benchmark.function
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
algorithm = nd.FiniteDifferenceFormula(benchmark.function, x)
approx_f3d = algorithm.compute_third_derivative(k)
print("Approx. f''(x) = ", approx_f3d)
exact_f3d = benchmark.third_derivative(x)
print("Exact f''(x) = ", exact_f3d)

relative_precision = 1.0e-14
print("relative_precision = ", relative_precision)
f3inf, f3sup = compute_f3_inf_sup(benchmark.function, x, k, relative_precision)
print("f3inf = ", f3inf)
print("f3sup = ", f3sup)

# %%
