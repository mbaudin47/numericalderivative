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
def my_exp(x):
    return np.exp(x)


# Define its exact derivative (for testing purposes only)
def my_exp_prime(x):
    return np.exp(x)


# Define its exact derivative (for testing purposes only)
def my_exp_3d_derivative(x):
    return np.exp(x)


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


class CheckStepleman(unittest.TestCase):
    def test_base_default_default_step(self):
        print("test_base_default_default_step")
        x = 1.0e0
        # Check approximate optimal h
        algorithm = nd.SteplemanWinarsky(my_exp, x, verbose=True)
        step_computed, iterations = algorithm.compute_step()
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        print("Optimum h =", step_computed)
        fdoptimal = nd.FiniteDifferenceOptimalStep()
        third_derivative_value = my_exp_3d_derivative(x)
        step_exact, absolute_error = fdoptimal.compute_step_first_derivative_central(
            third_derivative_value
        )
        print("step_exact = ", step_exact)
        print("iterations =", iterations)
        np.testing.assert_allclose(step_computed, step_exact, rtol=1.0e1)
        # Check approximate f'(x)
        f_prime_approx = algorithm.compute_first_derivative(step_computed)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = my_exp_prime(x)
        print("f_prime_exact = ", f_prime_exact)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print("Absolute error = ", absolute_error)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-15)

    def test_base(self):
        print("test_base")
        x = 1.0e0
        # Check approximate optimal h
        algorithm = nd.SteplemanWinarsky(my_scaled_exp, x, verbose=True)
        initial_step = 1.0e8
        step_computed, iterations = algorithm.compute_step(initial_step)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        print("Optimum h =", step_computed)
        fdoptimal = nd.FiniteDifferenceOptimalStep()
        third_derivative_value = my_scaled_exp_3d_derivative(x)
        step_exact, absolute_error = fdoptimal.compute_step_first_derivative_central(
            third_derivative_value
        )
        print("step_exact = ", step_exact)
        print("iterations =", iterations)
        np.testing.assert_allclose(step_computed, step_exact, atol=1.0e2)
        # Check approximate f'(x)
        f_prime_approx = algorithm.compute_first_derivative(step_computed)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = my_scaled_exp_prime(x)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print("Absolute error = ", absolute_error)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-15)

    def test_compute_step_with_bisection(self):
        print("test_compute_step_with_bisection")
        x = 1.0e0
        algorithm = nd.SteplemanWinarsky(my_scaled_exp, x, verbose=True)
        initial_h, number_of_iterations = algorithm.search_step_with_bisection(
            1.0e-10, 1.0e8,
        )
        print("number_of_iterations =", number_of_iterations)
        print("initial_h =", initial_h)
        fdoptimal = nd.FiniteDifferenceOptimalStep()
        third_derivative_value = my_scaled_exp_3d_derivative(x)
        step_exact, absolute_error = fdoptimal.compute_step_first_derivative_central(
            third_derivative_value
        )
        print("step_exact = ", step_exact)
        print("iterations =", number_of_iterations)
        np.testing.assert_allclose(initial_h, step_exact, atol=1.0e8)

    def test_compute_step_with_monotony(self):
        print("test_compute_step_with_monotony")
        x = 1.0e0
        h0 = 1.0e5
        algorithm = nd.SteplemanWinarsky(my_scaled_exp, x, verbose=True)
        initial_h, number_of_iterations = algorithm.compute_step(
            h0,
        )
        print("number_of_iterations =", number_of_iterations)
        print("initial_h =", initial_h)
        fdoptimal = nd.FiniteDifferenceOptimalStep()
        third_derivative_value = my_scaled_exp_3d_derivative(x)
        step_exact, absolute_error = fdoptimal.compute_step_first_derivative_central(
            third_derivative_value
        )
        print("step_exact = ", step_exact)
        print("iterations =", number_of_iterations)
        np.testing.assert_allclose(initial_h, step_exact, atol=1.0e8)


if __name__ == "__main__":
    unittest.main()
