#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Test for GillMurraySaundersWright class.
"""
import unittest
import numpy as np
import numericalderivative as nd


# Define a function
def my_scaled_exp(x):
    alpha = 1.0e6
    return np.exp(-x / alpha)


# Define its exact derivative (for testing purposes only)
def my_scaled_exp_prime(x):
    alpha = 1.0e6
    return -np.exp(-x / alpha) / alpha


# Define its exact derivative (for testing purposes only)
def my_scaled_exp_3d_derivative(x):
    alpha = 1.0e6
    return -np.exp(-x / alpha) / (alpha**3)


def scaled_exp_4th_derivative(x):
    alpha = 1.0e6
    return np.exp(-x / alpha) / (alpha**4)


class CheckGillMurraySaunders(unittest.TestCase):
    def test_base(self):
        # Check that the first derivative is correctly approximated
        print("test_base")
        x = 1.0e0
        relative_precision = 1.0e-15
        algorithm = nd.GillMurraySaundersWright(my_scaled_exp, x, relative_precision)
        kmin = 1.0e-2
        kmax = 1.0e7
        step, number_of_iterations = algorithm.compute_step(kmin, kmax)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Optimum h for f'=", step)
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        # Check optimal step
        fdoptimal = nd.FiniteDifferenceOptimalStep()
        third_derivative_value = my_scaled_exp_3d_derivative(x)
        exact_step, absolute_error = fdoptimal.compute_step_first_derivative_central(
            third_derivative_value
        )
        print("exact_step = ", exact_step)
        np.testing.assert_allclose(step, exact_step, atol=1.0e1)
        # Check f'(x)
        f_prime_approx = algorithm.compute_first_derivative(step)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = my_scaled_exp_prime(x)
        print("exact_f_prime_value = ", f_prime_exact)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print(f"Absolute error = {absolute_error:.3e}")
        relative_error = absolute_error / abs(f_prime_exact)
        print(f"Relative error = {relative_error:.3e}")
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-6)

    def test_second_derivative_step(self):
        # Check that the step for second derivative is OK
        x = 1.0e0
        relative_precision = 1.0e-15
        algorithm = nd.GillMurraySaundersWright(my_scaled_exp, x, relative_precision)
        kmin = 1.0e-2
        kmax = 1.0e7
        h_optimal_for_second_derivative, number_of_iterations = (
            algorithm.compute_step_for_second_derivative(kmin, kmax)
        )
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Optimum h for f''=", h_optimal_for_second_derivative)
        print("Function evaluations =", number_of_function_evaluations)
        fd_step = nd.FiniteDifferenceOptimalStep(relative_precision)
        fourth_derivative_value = scaled_exp_4th_derivative(x)
        k_optimal, absolute_error = fd_step.compute_step_second_derivative(
            fourth_derivative_value
        )
        print("Exact h for f''=", k_optimal)
        np.testing.assert_allclose(
            h_optimal_for_second_derivative, k_optimal, atol=1.0e3
        )

    def test_gms_exp_example_2nd_derivative(self):
        # This is (Gill, Murray, Saunders & Wright, 1983) example 1 page 312.
        # Check that the approximate optimal step for the second derivative is correctly approximated
        benchmark = nd.GillMurraySaundersWrightExponentialDerivativeBenchmark()
        relative_precision = 1.0e-15
        algorithm = nd.GillMurraySaundersWright(
            benchmark.function, benchmark.x, relative_precision, verbose=True
        )
        kmin = 1.0e-10
        kmax = 1.0e0
        h_optimal_for_second_derivative, number_of_iterations = (
            algorithm.compute_step_for_second_derivative(kmin, kmax)
        )
        print("Optimum h for f''=", h_optimal_for_second_derivative)
        print("Number of iterations=", number_of_iterations)
        fd_step = nd.FiniteDifferenceOptimalStep(relative_precision)
        fourth_derivative_value = scaled_exp_4th_derivative(benchmark.x)
        k_optimal, absolute_error = fd_step.compute_step_second_derivative(
            fourth_derivative_value
        )
        print("Exact h for f''=", k_optimal)
        np.testing.assert_allclose(
            h_optimal_for_second_derivative, k_optimal, atol=1.0e3
        )

    def test_gms_exp_example(self):
        # Check that the first derivative is correctly approximated
        benchmark = nd.GillMurraySaundersWrightExponentialDerivativeBenchmark()
        relative_precision = 1.0e-15
        algorithm = nd.GillMurraySaundersWright(
            benchmark.function, benchmark.x, relative_precision, verbose=True
        )
        kmin = 1.0e-10
        kmax = 1.0e0
        step, number_of_iterations = algorithm.compute_step(kmin, kmax)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Optimum h for f'=", step)
        print("Function evaluations =", number_of_function_evaluations)
        # Check optimal step
        fdoptimal = nd.FiniteDifferenceOptimalStep()
        third_derivative_value = benchmark.third_derivative(benchmark.x)
        exact_step, absolute_error = fdoptimal.compute_step_first_derivative_central(
            third_derivative_value
        )
        print("exact_step = ", exact_step)
        np.testing.assert_allclose(step, exact_step, atol=1.0e1)
        # Check f'(x)
        f_prime_approx = algorithm.compute_first_derivative(step)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = benchmark.first_derivative(benchmark.x)
        print("exact_f_prime_value = ", f_prime_exact)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print(f"Absolute error = {absolute_error:.3e}")
        relative_error = absolute_error / abs(f_prime_exact)
        print(f"Relative error = {relative_error:.3e}")
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
