#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Test for SteplemanWinarsky class.
"""
import unittest
import numpy as np
import numericalderivative as nd


# Define a function
def exp(x):
    return np.exp(x)


# Define its exact derivative (for testing purposes only)
def exp_prime(x):
    return np.exp(x)


# Define its exact third derivative (for testing purposes only)
def exp_3d_derivative(x):
    return np.exp(x)


# Define a function
def scaled_exp(x):
    alpha = 1.0e6
    return np.exp(-x / alpha)


# Define its exact derivative (for testing purposes only)
def scaled_exp_prime(x):
    alpha = 1.0e6
    return -np.exp(-x / alpha) / alpha


# Define its exact derivative (for testing purposes only)
def scaled_exp_3d_derivative(x):
    alpha = 1.0e6
    return -np.exp(-x / alpha) / (alpha**3)


class CheckStepleman(unittest.TestCase):
    def test_exp(self):
        print("test_exp")
        x = 1.0e0
        # Check approximate optimal h
        algorithm = nd.SteplemanWinarsky(exp, x, verbose=True)
        computed_step, iterations = algorithm.find_step()
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        print("Optimum h =", computed_step)
        third_derivative_value = exp_3d_derivative(x)
        exact_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value
        )
        print("exact_step = ", exact_step)
        print("iterations =", iterations)
        np.testing.assert_allclose(computed_step, exact_step, rtol=1.0e1)
        # Check approximate f'(x)
        f_prime_approx = algorithm.compute_first_derivative(computed_step)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = exp_prime(x)
        print("f_prime_exact = ", f_prime_exact)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print("Absolute error = ", absolute_error)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-15)

    def test_scaled_exp(self):
        print("test_scaled_exp")
        x = 1.0e0
        # Check approximate optimal h
        algorithm = nd.SteplemanWinarsky(scaled_exp, x, verbose=True)
        initial_step = 1.0e8
        computed_step, iterations = algorithm.find_step(initial_step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        print("Optimum h =", computed_step)
        third_derivative_value = scaled_exp_3d_derivative(x)
        exact_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value
        )
        print("exact_step = ", exact_step)
        print("iterations =", iterations)
        np.testing.assert_allclose(computed_step, exact_step, atol=1.0e2)
        # Check approximate f'(x)
        f_prime_approx = algorithm.compute_first_derivative(computed_step)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = scaled_exp_prime(x)
        print("f_prime_exact = ", f_prime_exact)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print("Absolute error = ", absolute_error)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-15)

    def test_compute_step_with_bisection(self):
        print("test_compute_step_with_bisection")
        x = 1.0e0
        algorithm = nd.SteplemanWinarsky(scaled_exp, x, verbose=True)
        initial_h, number_of_iterations = algorithm.find_initial_step(
            1.0e-10,
            1.0e8,
        )
        print("number_of_iterations =", number_of_iterations)
        print("initial_h =", initial_h)
        third_derivative_value = scaled_exp_3d_derivative(x)
        exact_step, _ = nd.FirstDerivativeCentral.compute_step(third_derivative_value)
        print("exact_step = ", exact_step)
        print("iterations =", number_of_iterations)
        np.testing.assert_allclose(initial_h, exact_step, atol=1.0e8)

    def test_sin_at_zero(self):
        """
        Consider f(x) = sin(x). At x = 0, we have f(x) = 0 and f'(x) = 1.
        Therefore, the algorithm must perform correctly.
        """
        print("test_sin_at_zero")
        x = 0.0e0
        # Check approximate optimal h
        algorithm = nd.SteplemanWinarsky(np.sin, x, verbose=True)
        initial_step = 1.0e0
        computed_step, iterations = algorithm.find_step(initial_step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        print("Optimum h =", computed_step)
        third_derivative_value = -np.cos(x)
        exact_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value
        )
        print("exact_step = ", exact_step)
        print("iterations =", iterations)
        np.testing.assert_allclose(computed_step, exact_step, atol=1.0e2)
        # Check approximate f'(x)
        f_prime_approx = algorithm.compute_first_derivative(computed_step)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = np.cos(x)
        print("f_prime_exact = ", f_prime_exact)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print("Absolute error = ", absolute_error)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, rtol=1.0e-7)

if __name__ == "__main__":
    unittest.main()
