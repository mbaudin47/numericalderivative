#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Use the benchmark problems
==========================

This example shows how to use a single benchmark problem or 
all the problems.

"""

# %%
import tabulate
import numericalderivative as nd
import math
import pylab as pl
import numpy as np

# %%
# First, we create an use a single problem.
# We create the problem and get the function and its
# first derivative

# %%
problem = nd.ExponentialProblem()
x = problem.get_x()
function = problem.get_function()
first_derivative = problem.get_first_derivative()

# %%
# Then we use a finite difference formula and compare it to the
# exact derivative.

# %%
formula = nd.FirstDerivativeForward(function, x)
step = 1.0e-5  # This is a first guess
approx_first_derivative = formula.compute(step)
exact_first_derivative = first_derivative(x)
absolute_error = abs(approx_first_derivative - exact_first_derivative)
print(f"Approximate first derivative = {approx_first_derivative}")
print(f"Exact first derivative = {exact_first_derivative}")
print(f"Absolute error = {absolute_error}")

# %%
# The problem is that the optimal step might not be the exact one.
# The optimal step can be computed using the second derivative, which is
# known in this problem.

# %%
second_derivative = problem.get_second_derivative()
second_derivative_value = second_derivative(x)
optimal_step_forward_formula, absolute_error = nd.FirstDerivativeForward.compute_step(
    second_derivative_value
)
print(f"Optimal step for forward derivative = {optimal_step_forward_formula}")
print(f"Minimum absolute error = {absolute_error}")

# %%
# Now use this step

# %%
approx_first_derivative = formula.compute(optimal_step_forward_formula)
exact_first_derivative = first_derivative(x)
absolute_error = abs(approx_first_derivative - exact_first_derivative)
print(f"Approximate first derivative = {approx_first_derivative}")
print(f"Exact first derivative = {exact_first_derivative}")
print(f"Absolute error = {absolute_error}")

# %%
# We can use a collection of benchmark problems.

# %%
benchmark = nd.BuildBenchmark()
number_of_problems = len(benchmark)
data = []
for i in range(number_of_problems):
    problem = benchmark[i]
    name = problem.get_name()
    x = problem.get_x()
    interval = problem.get_interval()
    data.append(
        [
            f"#{i} / {number_of_problems}",
            f"{name}",
            f"{x}",
            f"{interval[0]}",
            f"{interval[1]}",
        ]
    )

tabulate.tabulate(
    data,
    headers=["Index", "Name", "x", "xmin", "xmax"],
    tablefmt="html",
)

# %%
# Print each benchmark problems.

# %%
benchmark = nd.BuildBenchmark()
number_of_problems = len(benchmark)
for i in range(number_of_problems):
    problem = benchmark[i]
    print(problem)


# %%
# Plot the benchmark problems.

# %%
benchmark = nd.BuildBenchmark()
number_of_problems = len(benchmark)
number_of_columns = 3
number_of_rows = math.ceil(number_of_problems / number_of_columns)
number_of_points = 100
pl.figure(figsize=(8.0, 7.0))
data = []
index = 1
for i in range(number_of_problems):
    problem = benchmark[i]
    name = problem.get_name()
    print(f"Plot #{i}: {name}")
    x = problem.get_x()
    interval = problem.get_interval()
    function = problem.get_function()
    pl.subplot(number_of_rows, number_of_columns, index)
    x_grid = np.linspace(interval[0], interval[1], number_of_points)
    y_values = function(x_grid)
    pl.title(f"{name}")
    pl.plot(x_grid, y_values)
    # Update index
    index += 1

pl.subplots_adjust(wspace=0.5, hspace=1.2)

# %%
