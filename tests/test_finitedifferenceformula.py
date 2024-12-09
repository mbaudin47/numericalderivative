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


def scaled_exp_5th_derivative(x):
    alpha = 1.0e6
    return np.exp(-x / alpha) / (alpha**5)


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
        print(f"step = {step}, computed_absolute_error = {computed_absolute_error}")
        exact_absolute_error = nd.FirstDerivativeForward.compute_error(
            step, second_derivative_value, absolute_precision
        )
        np.testing.assert_allclose(
            computed_absolute_error, exact_absolute_error, rtol=1.0e-5
        )
        # Check that the step actually minimises the total error
        number_of_points = 10000
        step_array = np.logspace(-10.0, 5.0, number_of_points)
        model_error = [
            nd.FirstDerivativeForward.compute_error(
                step, second_derivative_value, absolute_precision
            )
            for step in step_array
        ]
        index = np.argmin(model_error)
        reference_optimal_step = step_array[index]
        reference_optimal_error = model_error[index]
        print(f"Optimal index = {index}")
        print(
            f"reference_optimal_step = {reference_optimal_step}, "
            f"reference_optimal_error = {reference_optimal_error}"
        )
        np.testing.assert_allclose(step, reference_optimal_step, rtol=1.0e-3)
        np.testing.assert_allclose(
            computed_absolute_error, reference_optimal_error, rtol=1.0e-3
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
        print(f"step = {step}, computed_absolute_error = {computed_absolute_error}")
        exact_absolute_error = nd.FirstDerivativeCentral.compute_error(
            step, third_derivative_value, absolute_precision
        )
        np.testing.assert_allclose(
            computed_absolute_error, exact_absolute_error, rtol=1.0e-5
        )
        # Check that the step actually minimises the total error
        number_of_points = 10000
        step_array = np.logspace(-10.0, 5.0, number_of_points)
        model_error = [
            nd.FirstDerivativeCentral.compute_error(
                step, third_derivative_value, absolute_precision
            )
            for step in step_array
        ]
        index = np.argmin(model_error)
        reference_optimal_step = step_array[index]
        reference_optimal_error = model_error[index]
        print(f"Optimal index = {index}")
        print(
            f"reference_optimal_step = {reference_optimal_step}, "
            f"reference_optimal_error = {reference_optimal_error}"
        )
        np.testing.assert_allclose(step, reference_optimal_step, rtol=2.0e-3)
        np.testing.assert_allclose(
            computed_absolute_error, reference_optimal_error, rtol=2.0e-3
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
        print(f"step = {step}, computed_absolute_error = {computed_absolute_error}")
        exact_absolute_error = nd.SecondDerivativeCentral.compute_error(
            step, fourth_derivative_value, absolute_precision
        )
        np.testing.assert_allclose(
            computed_absolute_error, exact_absolute_error, rtol=1.0e-5
        )
        # Check that the step actually minimises the total error
        number_of_points = 10000
        step_array = np.logspace(-10.0, 5.0, number_of_points)
        model_error = [
            nd.SecondDerivativeCentral.compute_error(
                step, fourth_derivative_value, absolute_precision
            )
            for step in step_array
        ]
        index = np.argmin(model_error)
        reference_optimal_step = step_array[index]
        reference_optimal_error = model_error[index]
        print(f"Optimal index = {index}")
        print(
            f"reference_optimal_step = {reference_optimal_step}, "
            f"reference_optimal_error = {reference_optimal_error}"
        )
        np.testing.assert_allclose(step, reference_optimal_step, rtol=1.0e-3)
        np.testing.assert_allclose(
            computed_absolute_error, reference_optimal_error, rtol=1.0e-3
        )

    def test_third_derivative(self):
        print("+ test_third_derivative")
        # Check ThirdDerivativeCentral.compute()
        x = 1.0
        absolute_precision = 1.0e-15
        fifth_derivative_value = scaled_exp_5th_derivative(x)
        print(f"fifth_derivative_value = {fifth_derivative_value}")
        step, absolute_error = nd.ThirdDerivativeCentral.compute_step(
            fifth_derivative_value, absolute_precision
        )
        finite_difference = nd.ThirdDerivativeCentral(scaled_exp, x)
        f_third_approx = finite_difference.compute(step)
        f_third_exact = scaled_exp_3d_derivative(x)
        print(
            f"step = {step}, "
            f"f_third_approx = {f_third_approx}, "
            f"f_third_exact = {f_third_exact}, "
            f"absolute_error = {absolute_error}"
        )
        np.testing.assert_allclose(f_third_approx, f_third_exact, atol=absolute_error)

    def test_third_derivative_step(self):
        print("+ test_third_derivative_step")
        # Check ThirdDerivativeCentral.compute_step()
        x = 1.0
        fifth_derivative_value = scaled_exp_5th_derivative(x)
        absolute_precision = 1.0e-15
        step, computed_absolute_error = nd.ThirdDerivativeCentral.compute_step(
            fifth_derivative_value, absolute_precision
        )
        print(f"step = {step}, computed_absolute_error = {computed_absolute_error}")
        exact_absolute_error = nd.ThirdDerivativeCentral.compute_error(
            step, fifth_derivative_value, absolute_precision
        )
        print(f"exact_absolute_error = {exact_absolute_error}")
        np.testing.assert_allclose(
            computed_absolute_error, exact_absolute_error, rtol=1.0e-5
        )
        # Check that the step actually minimises the total error
        number_of_points = 10000
        step_array = np.logspace(-10.0, 5.0, number_of_points)
        model_error = [
            nd.ThirdDerivativeCentral.compute_error(
                step, fifth_derivative_value, absolute_precision
            )
            for step in step_array
        ]
        index = np.argmin(model_error)
        reference_optimal_step = step_array[index]
        reference_optimal_error = model_error[index]
        print(f"Optimal index = {index}")
        print(
            f"reference_optimal_step = {reference_optimal_step}, "
            f"reference_optimal_error = {reference_optimal_error}"
        )
        np.testing.assert_allclose(step, reference_optimal_step, rtol=1.0e-3)
        np.testing.assert_allclose(
            computed_absolute_error, reference_optimal_error, rtol=1.0e-3
        )

if __name__ == "__main__":
    unittest.main()
