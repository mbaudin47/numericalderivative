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
def my_scaled_exp(x):
    alpha = 1.0e6
    return np.exp(-x / alpha)


def my_scaled_exp_prime(x):
    alpha = 1.0e6
    return -np.exp(-x / alpha) / alpha


def scaled_exp_2nd_derivative(x):
    alpha = 1.0e6
    return np.exp(-x / alpha) / (alpha**2)


def my_scaled_exp_3d_derivative(x):
    alpha = 1.0e6
    return -np.exp(-x / alpha) / (alpha**3)


def scaled_exp_4th_derivative(x):
    alpha = 1.0e6
    return np.exp(-x / alpha) / (alpha**4)


class CheckFiniteDifferenceFormula(unittest.TestCase):
    def test_first_derivative_forward(self):
        x = 1.0
        fd_optimal_step = nd.FiniteDifferenceOptimalStep()
        second_derivative_value = scaled_exp_2nd_derivative(x)
        step, absolute_error = fd_optimal_step.compute_step_first_derivative_forward(
            second_derivative_value
        )
        finite_difference = nd.FiniteDifferenceFormula(my_scaled_exp, x)
        f_prime_approx = finite_difference.compute_first_derivative_forward(step)
        f_prime_exact = my_scaled_exp_prime(x)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=absolute_error)

    def test_first_derivative_central(self):
        x = 1.0
        fd_optimal_step = nd.FiniteDifferenceOptimalStep()
        third_derivative_value = my_scaled_exp_3d_derivative(x)
        step, absolute_error = fd_optimal_step.compute_step_first_derivative_central(
            third_derivative_value
        )
        finite_difference = nd.FiniteDifferenceFormula(my_scaled_exp, x)
        f_prime_approx = finite_difference.compute_first_derivative_central(step)
        f_prime_exact = my_scaled_exp_prime(x)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=absolute_error)

    def test_second_derivative(self):
        x = 1.0
        fd_optimal_step = nd.FiniteDifferenceOptimalStep()
        fourth_derivative_value = scaled_exp_4th_derivative(x)
        step, absolute_error = fd_optimal_step.compute_step_second_derivative(
            fourth_derivative_value
        )
        finite_difference = nd.FiniteDifferenceFormula(my_scaled_exp, x)
        f_second_approx = finite_difference.compute_second_derivative_central(step)
        f_second_exact = scaled_exp_2nd_derivative(x)
        np.testing.assert_allclose(f_second_approx, f_second_exact, atol=absolute_error)


if __name__ == "__main__":
    unittest.main()
