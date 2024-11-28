#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Test for DumontetVignes class.
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


class CheckDumontetVignes(unittest.TestCase):
    def test_base(self):
        # h0 = 1.0e4
        x = 1.0e0
        # Check the step
        algorithm = nd.DumontetVignes(my_scaled_exp, x)
        h_optimal, number_of_iterations = algorithm.compute_step(
            kmin=1.0e-2,
            kmax=1.0e2,
        )
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Optimum h =", h_optimal)
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        fdoptimal = nd.FiniteDifferenceOptimalStep()
        third_derivative_value = my_scaled_exp_3d_derivative(x)
        exact_step, absolute_error = fdoptimal.compute_step_first_derivative_central(
            third_derivative_value
        )
        print("exact_step = ", exact_step)
        np.testing.assert_allclose(h_optimal, exact_step, atol=1.0e2)
        # Check f'
        f_prime_approx = algorithm.compute_first_derivative(h_optimal)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = my_scaled_exp_prime(x)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print("Absolute error = ", absolute_error)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-15)


if __name__ == "__main__":
    unittest.main()
