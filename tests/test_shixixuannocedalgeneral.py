#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Test for ShiXieXuanNocedalGeneral class.
"""
import unittest
import numpy as np
import numericalderivative as nd


# Define a function
def my_exp(x):
    return np.exp(x)


# Define its exact derivative (for testing purposes only)
def exp_prime(x):
    return np.exp(x)


# Define its exact seconde derivative (for testing purposes only)
def exp_2nd_derivative(x):
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


# Define its exact second derivative (for testing purposes only)
def scaled_exp_2nd_derivative(x):
    alpha = 1.0e6
    return np.exp(-x / alpha) / alpha**2


# Define its exact derivative (for testing purposes only)
def scaled_exp_3d_derivative(x):
    alpha = 1.0e6
    return -np.exp(-x / alpha) / (alpha**3)


class CheckShiXieXuanNocedalGeneral(unittest.TestCase):
    def test_base_default_default_step(self):
        print("test_base_default_default_step")
        x = 1.0e0
        differentiation_order = 1
        # Check approximate optimal h
        algorithm = nd.ShiXieXuanNocedalGeneral(my_exp, x, differentiation_order, verbose=True)
        initial_step = 1.0e0
        computed_step, number_of_iterations = algorithm.find_step(initial_step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        print("Optimum h =", computed_step)
        second_derivative_value = exp_2nd_derivative(x)
        step_exact, absolute_error = nd.FirstDerivativeForward.compute_step(
            second_derivative_value
        )
        print("step_exact = ", step_exact)
        print("number_of_iterations =", number_of_iterations)
        np.testing.assert_allclose(computed_step, step_exact, rtol=1.0e1)
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
        algorithm = nd.ShiXieXuanNocedalGeneral(scaled_exp, x, verbose=True)
        initial_step = 1.0e0
        computed_step, number_of_iterations = algorithm.find_step(initial_step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        print("Optimum h =", computed_step)
        second_derivative_value = scaled_exp_2nd_derivative(x)
        step_exact, absolute_error = nd.FirstDerivativeForward.compute_step(
            second_derivative_value
        )
        print("step_exact = ", step_exact)
        print("number_of_iterations =", number_of_iterations)
        np.testing.assert_allclose(computed_step, step_exact, atol=1.0e2)
        # Check approximate f'(x)
        f_prime_approx = algorithm.compute_first_derivative(computed_step)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = scaled_exp_prime(x)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print("Absolute error = ", absolute_error)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-15)

    def test_sin_at_zero(self):
        """
        Consider f(x) = sin(x). At x = 0, we have f(x) = 0 and f'(x) = cos(x) = cos(0) = 1.
        We have f''(x) = -sin(x) = -sin(0) = 0.
        Therefore, the algorithm must perform correctly.
        """
        print("test_sin_at_zero")
        x = 0.0e0
        # Check approximate optimal h
        algorithm = nd.ShiXieXuanNocedalGeneral(np.sin, x, verbose=True)
        initial_step = 1.0e0
        computed_step, number_of_iterations = algorithm.find_step(initial_step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        print("Optimum h =", computed_step)
        third_derivative_value = -np.cos(x)
        exact_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value
        )
        print("exact_step = ", exact_step)
        print("number_of_iterations =", number_of_iterations)
        np.testing.assert_allclose(computed_step, exact_step, atol=1.0e2)
        # Check approximate f'(x)
        f_prime_approx = algorithm.compute_first_derivative(computed_step)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = np.cos(x)
        print("f_prime_exact = ", f_prime_exact)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print("Absolute error = ", absolute_error)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, rtol=1.0e-7)

    def test_ratio(self):
        problem = nd.SinProblem()
        #
        function = problem.get_function()
        x = problem.get_x()
        #
        algorithm = nd.ShiXieXuanNocedalGeneral(function, x)
        absolute_precision = algorithm.get_absolute_precision()
        step = 1.0e-5
        test_ratio = algorithm.compute_test_ratio(step)
        print(f"test_ratio = {test_ratio}")
        second_derivative = problem.get_second_derivative()
        abs_second_derivative_value = abs(second_derivative(x))
        print(f"abs(f''(x)) = {abs_second_derivative_value}")
        #
        scaled_ratio = 4 * absolute_precision * test_ratio / (3 * step**2)
        print(f"scaled_ratio = {scaled_ratio}")
        relative_error = (
            abs(scaled_ratio - abs_second_derivative_value)
            / abs_second_derivative_value
        )
        print(f"Relative difference on scaled test ratio = {relative_error}")
        np.testing.assert_allclose(
            scaled_ratio, abs_second_derivative_value, rtol=1.0e-4
        )


if __name__ == "__main__":
    unittest.main()
