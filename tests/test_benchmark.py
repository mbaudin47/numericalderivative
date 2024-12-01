#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Test for DerivativeBenchmark class.
"""
import unittest
import numpy as np
import numericalderivative as nd

def check_benchmark_problem(benchmark):
    # Get fields
    name = benchmark.get_name()
    x = benchmark.get_x()
    function = benchmark.get_function()
    first_derivative = benchmark.get_first_derivative()
    second_derivative = benchmark.get_second_derivative()
    third_derivative = benchmark.get_third_derivative()
    fourth_derivative = benchmark.get_fourth_derivative()
    #
    fd_optimal_step = nd.FiniteDifferenceOptimalStep()
    finite_difference = nd.FiniteDifferenceFormula(function, x)
    #
    print(f"Check first derivative using second derivative for {name}")
    second_derivative_value = second_derivative(x)
    step, absolute_error = fd_optimal_step.compute_step_first_derivative_forward(
        second_derivative_value
    )
    f_prime_approx = finite_difference.compute_first_derivative_forward(step)
    f_prime_exact = first_derivative(x)
    print(
        f"({name}) f_prime_approx = {f_prime_approx}, "
        f"f_prime_exact = {f_prime_exact}"
    )
    np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=absolute_error)
    #
    print(f"Check first derivative using third derivative for {name}")
    third_derivative_value = third_derivative(x)
    step, absolute_error = fd_optimal_step.compute_step_first_derivative_central(
        third_derivative_value
    )
    f_prime_approx = finite_difference.compute_first_derivative_central(step)
    f_prime_exact = first_derivative(x)
    print(
        f"({name}) f_prime_approx = {f_prime_approx}, "
        f"f_prime_exact = {f_prime_exact}"
    )
    np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=absolute_error)
    #
    print(f"Check second derivative using fourth derivative for {name}")
    fourth_derivative_value = fourth_derivative(x)
    step, absolute_error = fd_optimal_step.compute_step_second_derivative(
        fourth_derivative_value
    )
    f_second_approx = finite_difference.compute_second_derivative_central(step)
    f_second_exact = second_derivative(x)
    print(
        f"({name}) f_second_approx = {f_second_approx}, "
        f"f_second_exact = {f_second_exact}"
    )
    np.testing.assert_allclose(f_second_approx, f_second_exact, atol=absolute_error)
    #
    print(f"Check third derivative for {name}")
    finite_difference = nd.FiniteDifferenceFormula(second_derivative, x)
    step = 1.0e-4
    f_third_approx = finite_difference.compute_first_derivative_central(step)
    f_third_exact = third_derivative(x)
    print(
        f"({name}) f_third_approx = {f_third_approx}, "
        f"f_third_exact = {f_third_exact}"
    )
    np.testing.assert_allclose(f_third_approx, f_third_exact, rtol=1.0e-4)
    #
    print(f"Check fourth derivative for {name}")
    finite_difference = nd.FiniteDifferenceFormula(third_derivative, x)
    step = 1.0e-4
    f_fourth_approx = finite_difference.compute_first_derivative_central(step)
    f_fourth_exact = fourth_derivative(x)
    print(
        f"({name}) f_fourth_approx = {f_fourth_approx}, "
        f"f_fourth_exact = {f_fourth_exact}"
    )
    np.testing.assert_allclose(f_fourth_approx, f_fourth_exact, rtol=1.0e-4)


class CheckDerivativeBenchmark(unittest.TestCase):
    def test_Exponential(self):
        benchmark = nd.ExponentialProblem()
        check_benchmark_problem(benchmark)

    def test_All(self):
        collection = nd.BuildBenchmark()
        for i in range(len(collection)):
            benchmark = collection[i]
            print(f"#{i}/{len(collection)}, checking {benchmark.name}")
            check_benchmark_problem(benchmark)


if __name__ == "__main__":
    unittest.main()
