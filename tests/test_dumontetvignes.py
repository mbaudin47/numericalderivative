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


class CheckDumontetVignes(unittest.TestCase):
    def test_base(self):
        # h0 = 1.0e4
        x = 1.0e0
        # Check the step
        algorithm = nd.DumontetVignes(scaled_exp, x)
        h_optimal, number_of_iterations = algorithm.find_step(
            kmin=1.0e-2,
            kmax=1.0e2,
        )
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Optimum h =", h_optimal)
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        assert number_of_iterations > 1
        third_derivative_value = scaled_exp_3d_derivative(x)
        exact_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value
        )
        print("exact_step = ", exact_step)
        np.testing.assert_allclose(h_optimal, exact_step, atol=1.0e2)
        # Check f'
        f_prime_approx = algorithm.compute_first_derivative(h_optimal)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = scaled_exp_prime(x)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print("Absolute error = ", absolute_error)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-15)

    def test_sin_at_zero(self):
        """
        Consider f(x) = sin(x). At x = 0, we have f(x) = 0 and f'(x) = cos(x) = cos(0) = 1.
        We have f''(x) = -sin(x) = -sin(0) = 0 and f'''(x) = -cos(x) = -cos(0) = -1.
        Since the third derivative is nonzero, the algorithm must perform correctly.
        """
        print("test_sin_at_zero")
        x = 0.0e0
        # Check approximate optimal h
        algorithm = nd.DumontetVignes(np.sin, x, verbose=True)
        computed_step, number_of_iterations = algorithm.find_step(
            kmin=1.0e-15, kmax=1.0e-1
        )
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

    def test_ell_ratio(self):
        problem = nd.SinProblem()
        #
        function = problem.get_function()
        x = problem.get_x()
        #
        algorithm = nd.DumontetVignes(function, x)
        step = 1.0e-3
        ell_ratio, f3inf, f3sup = algorithm.compute_ell(step)
        print(f"ell_ratio = {ell_ratio}, f3inf = {f3inf}, f3sup = {f3sup}")
        third_derivative = problem.get_third_derivative()
        third_derivative_value = third_derivative(x)
        print(f"f'''(x) = {third_derivative_value}")

        #
        relative_error = abs(f3inf - third_derivative_value) / abs(
            third_derivative_value
        )
        print(f"Relative difference on lower bound = {relative_error}")
        relative_error = abs(f3sup - third_derivative_value) / abs(
            third_derivative_value
        )
        print(f"Relative difference on lower bound = {relative_error}")
        np.testing.assert_allclose(f3inf, third_derivative_value, rtol=1.0e-5)
        np.testing.assert_allclose(f3sup, third_derivative_value, rtol=1.0e-5)
        np.testing.assert_allclose(ell_ratio, f3sup / f3inf, rtol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
