#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Test for DerivativeBenchmark class.
"""
import unittest
import numpy as np
import numericalderivative as nd


MaximumStep = 1.0e10


class ProblemChecker:
    def __init__(self, problem, tolerance_factor=2.0, number_of_points=50) -> None:
        self.problem = problem
        # Get fields
        self.name = problem.get_name()
        self.x = problem.get_x()
        self.function = problem.get_function()
        self.first_derivative = problem.get_first_derivative()
        self.second_derivative = problem.get_second_derivative()
        self.third_derivative = problem.get_third_derivative()
        self.fourth_derivative = problem.get_fourth_derivative()
        self.fifth_derivative = problem.get_fifth_derivative()
        self.interval = problem.get_interval()
        #
        self.tolerance_factor = tolerance_factor
        self.finite_difference = nd.FiniteDifferenceFormula(self.function, self.x)
        #
        self.is_first_derivative_from_third_enabled = True
        self.is_second_derivative_enabled = True
        self.is_second_derivative_from_first_enabled = True
        self.is_third_derivative_enabled = True
        self.is_fourth_derivative_enabled = True
        self.is_fifth_derivative_enabled = True
        self.number_of_points = number_of_points

    def check(self):
        self.test_function_evaluation()
        self.test_first_derivative_from_second()
        self.test_first_derivative_from_third()
        self.test_second_derivative()
        self.test_third_derivative()
        self.test_fourth_derivative()
        self.test_fifth_derivative()
        self.test_second_derivative_from_first()

    def test_function_evaluation(self):
        # Check the function evaluation on a grid
        x_grid = np.linspace(self.interval[0], self.interval[1], self.number_of_points)
        _ = self.function(x_grid)
        # This avoids to generate exceptions when evaluating the function.
        # Check that the test point is in the interval
        assert self.x >= self.interval[0]
        assert self.x <= self.interval[1]

    def skip_first_derivative_from_third(self):
        print("Skip first derivative from third test")
        self.is_first_derivative_from_third_enabled = False

    def skip_second_derivative(self):
        print("Skip second derivative test")
        self.is_second_derivative_enabled = False

    def skip_second_derivative_from_first(self):
        print("Skip second derivative from first test")
        self.is_second_derivative_from_first_enabled = False

    def skip_third_derivative(self):
        print("Skip third derivative test")
        self.is_third_derivative_enabled = False

    def skip_fourth_derivative(self):
        print("Skip fourth derivative test")
        self.is_fourth_derivative_enabled = False

    def skip_fifth_derivative(self):
        print("Skip fifth derivative test")
        self.is_fifth_derivative_enabled = False

    def test_first_derivative_from_second(self):
        print(f'Check first derivative using second derivative for "{self.name}"')
        second_derivative_value = self.second_derivative(self.x)
        step, absolute_error = nd.FirstDerivativeForward.compute_step(
            second_derivative_value
        )
        if step > MaximumStep:
            print(f"Warning: Step = {step} is larger than MaximumStep = {MaximumStep}.")
        f_prime_approx = nd.FirstDerivativeForward(self.function, self.x).compute(step)
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
        print(f'Check first derivative using third derivative for "{self.name}"')
        third_derivative_value = self.third_derivative(self.x)
        step, absolute_error = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value
        )
        if step > MaximumStep:
            print(f"Warning: Step = {step} is larger than MaximumStep = {MaximumStep}.")
        f_prime_approx = nd.FirstDerivativeCentral(self.function, self.x).compute(step)
        f_prime_exact = self.first_derivative(self.x)
        print(
            f"({self.name}) "
            f"third_derivative_value = {third_derivative_value}, "
            f"Step = {step:.4e}, absolute error = {absolute_error:.4e}, "
            f"f_prime_approx = {f_prime_approx}, "
            f"f_prime_exact = {f_prime_exact}"
        )
        if self.is_first_derivative_from_third_enabled:
            np.testing.assert_allclose(
                f_prime_approx,
                f_prime_exact,
                atol=self.tolerance_factor * absolute_error,
            )

    def test_second_derivative(self):
        print(f'Check second derivative using fourth derivative for "{self.name}"')
        fourth_derivative_value = self.fourth_derivative(self.x)
        step, absolute_error = nd.SecondDerivativeCentral.compute_step(
            fourth_derivative_value
        )
        f_second_approx = nd.SecondDerivativeCentral(self.function, self.x).compute(
            step
        )
        f_second_exact = self.second_derivative(self.x)
        print(
            f"({self.name}) "
            f"fourth_derivative_value = {fourth_derivative_value}, "
            f"Step = {step:.4e}, absolute error = {absolute_error:.4e}, "
            f"f_second_approx = {f_second_approx}, "
            f"f_second_exact = {f_second_exact}"
        )
        if self.is_second_derivative_enabled:
            np.testing.assert_allclose(
                f_second_approx,
                f_second_exact,
                atol=self.tolerance_factor * absolute_error,
            )

    def test_second_derivative_from_first(self):
        # The second derivative is the first derivative of the first derivative
        # (assuming the first derivative is OK)
        print(f'Check second derivative using first derivative for "{self.name}"')
        step = 1.0e-4
        f_second_approx = nd.FirstDerivativeCentral(
            self.first_derivative, self.x
        ).compute(step)
        f_second_exact = self.second_derivative(self.x)
        print(
            f"({self.name}) "
            f"Step = {step:.4e}, "
            f"f_second_approx = {f_second_approx}, "
            f"f_second_exact = {f_second_exact}"
        )
        if self.is_second_derivative_from_first_enabled:
            np.testing.assert_allclose(
                f_second_approx,
                f_second_exact,
                rtol=1.0e-3,
            )

    def test_third_derivative(self):
        print(f'Check third derivative for "{self.name}"')
        # The third derivative is the first derivative of the second derivative
        # (assuming the second derivative is OK)
        step = 1.0e-4
        f_third_approx = nd.FirstDerivativeCentral(
            self.second_derivative, self.x
        ).compute(step)
        f_third_exact = self.third_derivative(self.x)
        print(
            f"({self.name}) step = {step:.4e}, "
            f"f_third_approx = {f_third_approx}, "
            f"f_third_exact = {f_third_exact}"
        )
        if self.is_third_derivative_enabled:
            np.testing.assert_allclose(
                f_third_approx, f_third_exact, rtol=self.tolerance_factor * 1.0e-4
            )

    def test_fourth_derivative(self):
        print(f'Check fourth derivative for "{self.name}"')
        # The fourth derivative is the first derivative of the third derivative
        # (assuming the third derivative is OK)
        step = 1.0e-4
        f_fourth_approx = nd.FirstDerivativeCentral(
            self.third_derivative, self.x
        ).compute(step)
        f_fourth_exact = self.fourth_derivative(self.x)
        print(
            f"({self.name}) step = {step:.4e}, "
            f"f_fourth_approx = {f_fourth_approx}, "
            f"f_fourth_exact = {f_fourth_exact}"
        )
        if self.is_fourth_derivative_enabled:
            np.testing.assert_allclose(
                f_fourth_approx, f_fourth_exact, rtol=self.tolerance_factor * 1.0e-4
            )

    def test_fifth_derivative(self):
        print(f'Check fifth derivative for "{self.name}"')
        # The fifth derivative is the second derivative of the third derivative
        # (assuming the third derivative is OK)
        step = 1.0e-4
        f_fifth_approx = nd.SecondDerivativeCentral(
            self.third_derivative, self.x
        ).compute(step)
        f_fifth_exact = self.fifth_derivative(self.x)
        print(
            f"({self.name}) step = {step:.4e}, "
            f"f_fifth_approx = {f_fifth_approx}, "
            f"f_fifth_exact = {f_fifth_exact}"
        )
        if self.is_fifth_derivative_enabled:
            np.testing.assert_allclose(
                f_fifth_approx, f_fifth_exact, rtol=self.tolerance_factor * 1.0e-4
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
                # This test cannot pass the second derivative test: the fourth
                # derivative is zero, which produces an infinite optimal second
                # derivative step for central finite difference formula.
                checker.skip_second_derivative()
            elif name == "polynomial":
                # This test cannot pass the first derivative from third test:
                # the third derivative is zero, which produces an infinite
                # optimal first derivative step for central finite difference formula.
                checker.skip_first_derivative_from_third()
                checker.skip_second_derivative()
            elif name == "scaled exp":
                # Skip the fifth derivative test: it is too close to zero.
                checker.skip_fifth_derivative()
            elif name == "GMSW":
                # Skip the fifth derivative test: not implemented
                checker.skip_fifth_derivative()

            checker.check()
        print(f"Total = {len(collection)} problems.")


if __name__ == "__main__":
    unittest.main()
