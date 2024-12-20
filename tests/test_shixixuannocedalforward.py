#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Test for ShiXieXuanNocedalForward class.
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


class CheckShiXieXuanNocedalForward(unittest.TestCase):
    def test_base_default_default_step(self):
        print("test_base_default_default_step")
        x = 1.0e0
        # Check approximate optimal h
        algorithm = nd.ShiXieXuanNocedalForward(my_exp, x, verbose=True)
        initial_step = 1.0e0
        step_computed, iterations = algorithm.compute_step(initial_step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        print("Optimum h =", step_computed)
        second_derivative_value = exp_2nd_derivative(x)
        step_exact, absolute_error = nd.FirstDerivativeForward.compute_step(
            second_derivative_value
        )
        print("step_exact = ", step_exact)
        print("iterations =", iterations)
        np.testing.assert_allclose(step_computed, step_exact, rtol=1.0e1)
        # Check approximate f'(x)
        f_prime_approx = algorithm.compute_first_derivative(step_computed)
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
        algorithm = nd.ShiXieXuanNocedalForward(scaled_exp, x, verbose=True)
        initial_step = 1.0e0
        step_computed, iterations = algorithm.compute_step(initial_step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        print("Optimum h =", step_computed)
        second_derivative_value = scaled_exp_2nd_derivative(x)
        step_exact, absolute_error = nd.FirstDerivativeForward.compute_step(
            second_derivative_value
        )
        print("step_exact = ", step_exact)
        print("iterations =", iterations)
        np.testing.assert_allclose(step_computed, step_exact, atol=1.0e2)
        # Check approximate f'(x)
        f_prime_approx = algorithm.compute_first_derivative(step_computed)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = scaled_exp_prime(x)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print("Absolute error = ", absolute_error)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-15)


if __name__ == "__main__":
    unittest.main()
