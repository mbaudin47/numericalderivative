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


def scaled_exp_4th_derivative(x):
    alpha = 1.0e6
    return np.exp(-x / alpha) / (alpha**4)


class CheckGillMurraySaunders(unittest.TestCase):
    def test_base(self):
        # Check that the first derivative is correctly approximated
        print("+ test_base")
        x = 1.0e0
        absolute_precision = 1.0e-15
        algorithm = nd.GillMurraySaundersWright(scaled_exp, x, absolute_precision)
        kmin = 1.0e-2
        kmax = 1.0e7
        step, number_of_iterations = algorithm.find_step(kmin, kmax)
        assert number_of_iterations > 0
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Optimum h for f'=", step)
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        # Check optimal step
        third_derivative_value = scaled_exp_3d_derivative(x)
        exact_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value
        )
        print("exact_step = ", exact_step)
        np.testing.assert_allclose(step, exact_step, atol=1.0e1)
        # Check f'(x)
        f_prime_approx = algorithm.compute_first_derivative(step)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = scaled_exp_prime(x)
        print("exact_f_prime_value = ", f_prime_exact)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print(f"Absolute error = {absolute_error:.3e}")
        relative_error = absolute_error / abs(f_prime_exact)
        print(f"Relative error = {relative_error:.3e}")
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-6)

    def test_second_derivative_step(self):
        print("+ test_second_derivative_step")
        # Check that the step for second derivative is OK
        x = 1.0e0
        absolute_precision = 1.0e-15
        algorithm = nd.GillMurraySaundersWright(scaled_exp, x, absolute_precision)
        kmin = 1.0e-2
        kmax = 1.0e7
        h_optimal_for_second_derivative, number_of_iterations = (
            algorithm.compute_step_for_second_derivative(kmin, kmax)
        )
        assert number_of_iterations > 0
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Optimum h for f''=", h_optimal_for_second_derivative)
        print("Function evaluations =", number_of_function_evaluations)
        fourth_derivative_value = scaled_exp_4th_derivative(x)
        k_optimal, _ = nd.SecondDerivativeCentral.compute_step(fourth_derivative_value)
        print("Exact h for f''=", k_optimal)
        np.testing.assert_allclose(
            h_optimal_for_second_derivative, k_optimal, atol=1.0e3
        )

    def test_gms_exp_example_2nd_derivative(self):
        # This is (Gill, Murray, Saunders & Wright, 1983) example 1 page 312.
        # Check that the approximate optimal step for the second derivative is correctly approximated
        problem = nd.GMSWExponentialProblem()
        absolute_precision = 1.0e-15
        algorithm = nd.GillMurraySaundersWright(
            problem.get_function(), problem.get_x(), absolute_precision, verbose=True
        )
        kmin = 1.0e-10
        kmax = 1.0e0
        h_optimal_for_second_derivative, number_of_iterations = (
            algorithm.compute_step_for_second_derivative(kmin, kmax)
        )
        print("Optimum h for f''=", h_optimal_for_second_derivative)
        print("Number of iterations=", number_of_iterations)
        fourth_derivative_value = scaled_exp_4th_derivative(problem.get_x())
        k_optimal, _ = nd.SecondDerivativeCentral.compute_step(fourth_derivative_value)
        print("Exact h for f''=", k_optimal)
        np.testing.assert_allclose(
            h_optimal_for_second_derivative, k_optimal, atol=1.0e3
        )

    def test_gms_exp_example(self):
        # Check that the first derivative is correctly approximated
        problem = nd.GMSWExponentialProblem()
        absolute_precision = 1.0e-15
        algorithm = nd.GillMurraySaundersWright(
            problem.get_function(), problem.get_x(), absolute_precision, verbose=True
        )
        kmin = 1.0e-10
        kmax = 1.0e0
        step, _ = algorithm.find_step(kmin, kmax)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Optimum h for f'=", step)
        print("Function evaluations =", number_of_function_evaluations)
        # Check optimal step
        third_derivative_value = problem.third_derivative(problem.get_x())
        exact_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value
        )
        print("exact_step = ", exact_step)
        np.testing.assert_allclose(step, exact_step, atol=1.0e1)
        # Check f'(x)
        f_prime_approx = algorithm.compute_first_derivative(step)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = problem.first_derivative(problem.get_x())
        print("exact_f_prime_value = ", f_prime_exact)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print(f"Absolute error = {absolute_error:.3e}")
        relative_error = absolute_error / abs(f_prime_exact)
        print(f"Relative error = {relative_error:.3e}")
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, atol=1.0e-6)

    def test_expm1_at_zero(self):
        """
        Consider f(x) = exp(x) - 1. At x = 0, we have f(x) = 0 and f'(x) = 1.
        Moreover, the second derivative is nonzero: f''(x) = 1.
        Therefore, the algorithm must perform correctly.
        """
        print("+ test_expm1_at_zero")
        x = 0.0e0
        # Check approximate optimal h
        algorithm = nd.GillMurraySaundersWright(np.expm1, x, verbose=True)
        kmin = 1.0e-10
        kmax = 1.0e0
        computed_step, number_of_iterations = algorithm.find_step(kmin, kmax)
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        print("Function evaluations =", number_of_function_evaluations)
        assert number_of_function_evaluations > 0
        print("Optimum h =", computed_step)
        third_derivative_value = -np.cos(x)
        exact_step, absolute_error = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value
        )
        print("exact_step = ", exact_step)
        print("iterations =", number_of_iterations)
        np.testing.assert_allclose(computed_step, exact_step, atol=1.0e2)
        # Check approximate f'(x)
        f_prime_approx = algorithm.compute_first_derivative(computed_step)
        print("f_prime_approx = ", f_prime_approx)
        f_prime_exact = np.exp(x)
        print("f_prime_exact = ", f_prime_exact)
        absolute_error = abs(f_prime_approx - f_prime_exact)
        print("Absolute error = ", absolute_error)
        np.testing.assert_allclose(f_prime_approx, f_prime_exact, rtol=1.0e-7)

    def test_condition(self):
        problem = nd.SinProblem()
        function = problem.get_function()
        x = problem.get_x()
        #
        algorithm = nd.GillMurraySaundersWright(function, x)
        absolute_precision = algorithm.get_absolute_precision()
        step = 1.0e-5
        condition = algorithm.compute_condition(step)
        print(f"condition = {condition}")
        second_derivative = problem.get_second_derivative()
        abs_second_derivative_value = abs(second_derivative(x))
        print(f"abs(f''(x)) = {abs_second_derivative_value}")
        #
        scaled_condition = 4 * absolute_precision / (condition * step**2)
        print(f"scaled_condition = {scaled_condition}")
        #
        relative_error = (
            abs(scaled_condition - abs_second_derivative_value)
            / abs_second_derivative_value
        )
        print(f"Relative difference on scaled condition = {relative_error}")
        np.testing.assert_allclose(
            scaled_condition, abs_second_derivative_value, rtol=1.0e-6
        )


if __name__ == "__main__":
    unittest.main()
