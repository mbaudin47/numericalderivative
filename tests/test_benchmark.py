#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Test for DerivativeBenchmark class.
"""
import unittest
import numpy as np
import numericalderivative as nd


def check_benchmark(benchmark):
    x = benchmark.x
    fd_optimal_step = nd.FiniteDifferenceOptimalStep()
    finite_difference = nd.FiniteDifferenceFormula(benchmark.function, x)
    #
    print(f"Check first derivative using second derivative for {benchmark.name}")
    second_derivative_value = benchmark.second_derivative(x)
    step, absolute_error = fd_optimal_step.compute_step_first_derivative_forward(
        second_derivative_value
    )
    f_prime_approx = finite_difference.compute_first_derivative_forward(step)
    f_prime_exact = benchmark.first_derivative(x)
    print(
        f"({benchmark.name}) f_prime_approx = {f_prime_approx}, "
        f"f_prime_exact = {f_prime_exact}"
    )
    np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=absolute_error)
    #
    print(f"Check first derivative using third derivative for {benchmark.name}")
    third_derivative_value = benchmark.third_derivative(x)
    step, absolute_error = fd_optimal_step.compute_step_first_derivative_central(
        third_derivative_value
    )
    f_prime_approx = finite_difference.compute_first_derivative_central(step)
    f_prime_exact = benchmark.first_derivative(x)
    print(
        f"({benchmark.name}) f_prime_approx = {f_prime_approx}, "
        f"f_prime_exact = {f_prime_exact}"
    )
    np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=absolute_error)
    #
    print(f"Check second derivative using fourth derivative for {benchmark.name}")
    fourth_derivative_value = benchmark.fourth_derivative(x)
    step, absolute_error = fd_optimal_step.compute_step_second_derivative(
        fourth_derivative_value
    )
    f_second_approx = finite_difference.compute_second_derivative_central(step)
    f_second_exact = benchmark.second_derivative(x)
    print(
        f"({benchmark.name}) f_second_approx = {f_second_approx}, "
        f"f_second_exact = {f_second_exact}"
    )
    np.testing.assert_allclose(f_second_approx, f_second_exact, atol=absolute_error)
    #
    print(f"Check third derivative for {benchmark.name}")
    finite_difference = nd.FiniteDifferenceFormula(benchmark.second_derivative, x)
    step = 1.0e-4
    f_third_approx = finite_difference.compute_first_derivative_central(step)
    f_third_exact = benchmark.third_derivative(x)
    print(
        f"({benchmark.name}) f_third_approx = {f_third_approx}, "
        f"f_third_exact = {f_third_exact}"
    )
    np.testing.assert_allclose(f_third_approx, f_third_exact, rtol=1.0e-4)
    #
    print(f"Check fourth derivative for {benchmark.name}")
    finite_difference = nd.FiniteDifferenceFormula(benchmark.third_derivative, x)
    step = 1.0e-4
    f_fourth_approx = finite_difference.compute_first_derivative_central(step)
    f_fourth_exact = benchmark.fourth_derivative(x)
    print(
        f"({benchmark.name}) f_fourth_approx = {f_fourth_approx}, "
        f"f_fourth_exact = {f_fourth_exact}"
    )
    np.testing.assert_allclose(f_fourth_approx, f_fourth_exact, rtol=1.0e-4)


class CheckDerivativeBenchmark(unittest.TestCase):
    def test_Exponential(self):
        benchmark = nd.ExponentialDerivativeBenchmark()
        check_benchmark(benchmark)

    def test_All(self):
        collection = nd.BuildBenchmarkList()
        for i in range(len(collection)):
            benchmark = collection[i]
            print(f"Checking {benchmark.name}")
            check_benchmark(benchmark)


if __name__ == "__main__":
    unittest.main()
