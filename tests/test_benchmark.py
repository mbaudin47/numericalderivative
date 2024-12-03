#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Test for DerivativeBenchmark class.
"""
import unittest
import numpy as np
import numericalderivative as nd


class ProblemChecker:
    def __init__(self, problem, tolerance_factor=2.0) -> None:
        self.problem = problem
        # Get fields
        self.name = problem.get_name()
        self.x = problem.get_x()
        self.function = problem.get_function()
        self.first_derivative = problem.get_first_derivative()
        self.second_derivative = problem.get_second_derivative()
        self.third_derivative = problem.get_third_derivative()
        self.fourth_derivative = problem.get_fourth_derivative()
        #
        self.tolerance_factor = tolerance_factor
        self.fd_optimal_step = nd.FiniteDifferenceOptimalStep()
        self.finite_difference = nd.FiniteDifferenceFormula(self.function, self.x)
        #
        self.check_second_derivative = True
        self.check_third_derivative = True
        self.check_fourth_derivative = True

    def check(self):
        self.test_first_derivative_from_second()
        self.test_first_derivative_from_third()
        if self.check_second_derivative:
            self.test_second_derivative()
        if self.check_third_derivative:
            self.test_third_derivative()
        if self.check_fourth_derivative:
            self.test_fourth_derivative()

    def skip_second_derivative(self):
        self.check_second_derivative = False

    def skip_third_derivative(self):
        self.check_third_derivative = False

    def skip_fourth_derivative(self):
        self.check_fourth_derivative = False

    def test_first_derivative_from_second(self):
        print(f"Check first derivative using second derivative for {self.name}")
        second_derivative_value = self.second_derivative(self.x)
        step, absolute_error = (
            self.fd_optimal_step.compute_step_first_derivative_forward(
                second_derivative_value
            )
        )
        f_prime_approx = self.finite_difference.compute_first_derivative_forward(step)
        f_prime_exact = self.first_derivative(self.x)
        print(
            f"({self.name}) "
            f"second_derivative_value = {second_derivative_value}, "
            f"Step = {step:.4e}, absolute error = {absolute_error:.4e}, "
            f"f_prime_approx = {f_prime_approx}, "
            f"f_prime_exact = {f_prime_exact}"
        )
        np.testing.assert_allclose(
            f_prime_approx, f_prime_exact, atol=self.tolerance_factor * absolute_error
        )

    def test_first_derivative_from_third(self):
        print(f"Check first derivative using third derivative for {self.name}")
        third_derivative_value = self.third_derivative(self.x)
        step, absolute_error = (
            self.fd_optimal_step.compute_step_first_derivative_central(
                third_derivative_value
            )
        )
        f_prime_approx = self.finite_difference.compute_first_derivative_central(step)
        f_prime_exact = self.first_derivative(self.x)
        print(
            f"({self.name}) "
            f"third_derivative_value = {third_derivative_value}, "
            f"Step = {step:.4e}, absolute error = {absolute_error:.4e}, "
            f"f_prime_approx = {f_prime_approx}, "
            f"f_prime_exact = {f_prime_exact}"
        )
        np.testing.assert_allclose(
            f_prime_approx, f_prime_exact, atol=self.tolerance_factor * absolute_error
        )

    def test_second_derivative(self):
        print(f"Check second derivative using fourth derivative for {self.name}")
        fourth_derivative_value = self.fourth_derivative(self.x)
        step, absolute_error = self.fd_optimal_step.compute_step_second_derivative(
            fourth_derivative_value
        )
        f_second_approx = self.finite_difference.compute_second_derivative_central(step)
        f_second_exact = self.second_derivative(self.x)
        print(
            f"({self.name}) "
            f"fourth_derivative_value = {fourth_derivative_value}, "
            f"Step = {step:.4e}, absolute error = {absolute_error:.4e}, "
            f"f_second_approx = {f_second_approx}, "
            f"f_second_exact = {f_second_exact}"
        )
        np.testing.assert_allclose(
            f_second_approx, f_second_exact, atol=self.tolerance_factor * absolute_error
        )

    def test_third_derivative(self):
        print(f"Check third derivative for {self.name}")
        finite_difference = nd.FiniteDifferenceFormula(self.second_derivative, self.x)
        step = 1.0e-4
        f_third_approx = finite_difference.compute_first_derivative_central(step)
        f_third_exact = self.third_derivative(self.x)
        print(
            f"({self.name}) step = {step:.4e}, "
            f"f_third_approx = {f_third_approx}, "
            f"f_third_exact = {f_third_exact}"
        )
        np.testing.assert_allclose(
            f_third_approx, f_third_exact, rtol=self.tolerance_factor * 1.0e-4
        )

    def test_fourth_derivative(self):
        print(f"Check fourth derivative for {self.name}")
        finite_difference = nd.FiniteDifferenceFormula(self.third_derivative, self.x)
        step = 1.0e-4
        f_fourth_approx = finite_difference.compute_first_derivative_central(step)
        f_fourth_exact = self.fourth_derivative(self.x)
        print(
            f"({self.name}) step = {step:.4e}, "
            f"f_fourth_approx = {f_fourth_approx}, "
            f"f_fourth_exact = {f_fourth_exact}"
        )
        np.testing.assert_allclose(
            f_fourth_approx, f_fourth_exact, rtol=self.tolerance_factor * 1.0e-4
        )


class CheckDerivativeBenchmark(unittest.TestCase):
    def test_Exponential(self):
        problem = nd.ExponentialProblem()
        checker = ProblemChecker(problem)
        checker.check()

    def test_All(self):
        collection = nd.BuildBenchmark()
        for i in range(len(collection)):
            problem = collection[i]
            name = problem.get_name()
            print(f"#{i}/{len(collection)}, checking {name}")
            checker = ProblemChecker(problem)
            if name == "SXXN4":
                # This test cannot pass: the fourth derivative
                # is zero, which produces an infinite optimal second derivative step
                # for central finite difference formula.
                checker.skip_second_derivative()
            checker.check()
        print(f"Total = {len(collection)} problems.")

if __name__ == "__main__":
    unittest.main()
