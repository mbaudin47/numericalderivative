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
def scaled_exp(x):
    alpha = 1.0e6
    return np.exp(-x / alpha)


def scaled_exp_prime(x):
    alpha = 1.0e6
    return -np.exp(-x / alpha) / alpha


def scaled_exp_2nd_derivative(x):
    alpha = 1.0e6
    return np.exp(-x / alpha) / (alpha**2)


def scaled_exp_3d_derivative(x):
    alpha = 1.0e6
    return -np.exp(-x / alpha) / (alpha**3)


def scaled_exp_4th_derivative(x):
    alpha = 1.0e6
    return np.exp(-x / alpha) / (alpha**4)


class CheckFiniteDifferenceFormula(unittest.TestCase):
    def test_first_derivative_forward(self):
        print("+ test_first_derivative_forward")
        # Check FirstDerivativeForward.compute()
        x = 1.0
        second_derivative_value = scaled_exp_2nd_derivative(x)
        step, absolute_error = nd.FirstDerivativeForward.compute_step(
            second_derivative_value
        )
        finite_difference = nd.FirstDerivativeForward(scaled_exp, x)
        f_prime_approx = finite_difference.compute(step)
        f_prime_exact = scaled_exp_prime(x)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=absolute_error)

    def test_first_derivative_forward_error(self):
        print("+ test_first_derivative_forward_error")
        # Check FirstDerivativeForward.compute_step
        x = 1.0
        absolute_precision = 1.0e-15
        second_derivative_value = scaled_exp_2nd_derivative(x)
        step, computed_absolute_error = nd.FirstDerivativeForward.compute_step(
            second_derivative_value, absolute_precision
        )
        exact_absolute_error = nd.FirstDerivativeForward.compute_error(
            step, second_derivative_value, absolute_precision
        )
        np.testing.assert_allclose(
            computed_absolute_error, exact_absolute_error, rtol=1.0e-5
        )

    def test_first_derivative_central(self):
        print("+ test_first_derivative_central")
        # Check FirstDerivativeCentral.compute()
        x = 1.0
        third_derivative_value = scaled_exp_3d_derivative(x)
        step, absolute_error = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value
        )
        finite_difference = nd.FirstDerivativeCentral(scaled_exp, x)
        f_prime_approx = finite_difference.compute(step)
        f_prime_exact = scaled_exp_prime(x)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=absolute_error)

    def test_first_derivative_central_error(self):
        print("+ test_first_derivative_central_error")
        # Check nd.FirstDerivativeCentral.compute_step
        x = 1.0
        third_derivative_value = scaled_exp_3d_derivative(x)
        absolute_precision = 1.0e-15
        step, computed_absolute_error = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value, absolute_precision
        )
        exact_absolute_error = nd.FirstDerivativeCentral.compute_error(
            step, third_derivative_value, absolute_precision
        )
        np.testing.assert_allclose(
            computed_absolute_error, exact_absolute_error, rtol=1.0e-5
        )

    def test_second_derivative(self):
        print("+ test_second_derivative")
        # Check SecondDerivativeCentral.compute()
        x = 1.0
        fourth_derivative_value = scaled_exp_4th_derivative(x)
        step, absolute_error = nd.SecondDerivativeCentral.compute_step(
            fourth_derivative_value
        )
        finite_difference = nd.SecondDerivativeCentral(scaled_exp, x)
        f_second_approx = finite_difference.compute(step)
        f_second_exact = scaled_exp_2nd_derivative(x)
        np.testing.assert_allclose(f_second_approx, f_second_exact, atol=absolute_error)

    def test_second_derivative_step(self):
        print("+ test_second_derivative_step")
        # Check SecondDerivativeCentral.compute_step()
        x = 1.0
        fourth_derivative_value = scaled_exp_4th_derivative(x)
        absolute_precision = 1.0e-15
        step, computed_absolute_error = nd.SecondDerivativeCentral.compute_step(
            fourth_derivative_value, absolute_precision
        )
        exact_absolute_error = nd.SecondDerivativeCentral.compute_error(
            step, fourth_derivative_value, absolute_precision
        )
        np.testing.assert_allclose(
            computed_absolute_error, exact_absolute_error, rtol=1.0e-5
        )

    def test_third_derivative(self):
        print("+ test_third_derivative")
        # Check ThirdDerivativeCentral.compute()
        x = 1.0
        step = 1.0e-3
        finite_difference = nd.ThirdDerivativeCentral(np.exp, x)
        f_third_approx = finite_difference.compute(step)
        f_third_exact = np.exp(x)
        print(f"f_third_approx = {f_third_approx}, f_third_exact = {f_third_exact}")
        np.testing.assert_allclose(f_third_approx, f_third_exact, rtol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
