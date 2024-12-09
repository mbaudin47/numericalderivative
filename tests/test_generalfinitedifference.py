#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Test for GeneralFiniteDifference class.
"""
import unittest
import numpy as np
import numericalderivative as nd


class CheckGeneralFD(unittest.TestCase):
    def test_finite_differences(self):
        print("+ test_finite_differences")
        # Check finite_differences
        # Evalue f'''(x) with f(x)= sin(x)
        problem = nd.SinProblem()
        function = problem.get_function()
        function_third_derivative = problem.get_third_derivative()
        x = 1.0
        f_third_derivative_exact = function_third_derivative(x)
        print(f"f_third_derivative_exact = {f_third_derivative_exact}")
        differentiation_order = 3
        # centered formula is for even accuracy
        direction = "central"
        for formula_accuracy in [2, 4, 6]:
            formula = nd.GeneralFiniteDifference(
                function,
                x,
                differentiation_order,
                formula_accuracy,
                direction=direction,
            )
            step, _ = formula.compute_step()
            f_third_derivative_approx = formula.compute(step)
            absolute_error = abs(f_third_derivative_exact - f_third_derivative_approx)
            print(
                f"formula_accuracy = {formula_accuracy}, "
                f"step = {step:.4e}, "
                f"f_third_derivative_approx = {f_third_derivative_approx}, "
                f"absolute_error = {absolute_error:.4e}"
            )
            np.testing.assert_allclose(
                f_third_derivative_approx, f_third_derivative_exact, rtol=1.0e-5
            )
        # forward and backward formula are ok for even accuracy
        for formula_accuracy in range(3, 5):
            for direction in ["forward", "backward"]:
                formula = nd.GeneralFiniteDifference(
                    function,
                    x,
                    differentiation_order,
                    formula_accuracy,
                    direction=direction,
                )
                step, _ = formula.compute_step()
                f_third_derivative_approx = formula.compute(step)
                absolute_error = abs(
                    f_third_derivative_exact - f_third_derivative_approx
                )
                print(
                    f"formula_accuracy = {formula_accuracy}, "
                    f"step = {step:.4e}, "
                    f"f_third_derivative_approx = {f_third_derivative_approx}, "
                    f"absolute_error = {absolute_error:.4e}"
                )
                np.testing.assert_allclose(
                    f_third_derivative_approx, f_third_derivative_exact, rtol=1.0e-5
                )

    def test_first_forward(self):
        print("+ test_first_forward")
        # Evaluate f'(x) with f(x)= sin(x) using forward F.D.
        problem = nd.ScaledExponentialProblem()
        function = problem.get_function()
        function_second_derivative = problem.get_second_derivative()
        x = problem.get_x()
        differentiation_order = 1
        formula_accuracy = 1
        formula = nd.GeneralFiniteDifference(
            function,
            x,
            differentiation_order,
            formula_accuracy,
            direction="forward",
        )
        # Check indices
        imin, imax = formula.get_indices_min_max()
        assert imin == 0
        assert imax == 1
        # Check coefficients
        coefficients = formula.get_coefficients()
        np.testing.assert_allclose(coefficients, [-1.0, 1.0], rtol=1.0e-5)
        # Compute step and absolute error
        absolute_precision = 1.0e-16
        second_derivative_value = function_second_derivative(x)
        computed_step, computed_absolute_error = formula.compute_step(
            second_derivative_value, absolute_precision
        )
        print(
            f"Computed step = {computed_step:.4e}, "
            f"computed_absolute_error = {computed_absolute_error:.4e}"
        )
        # Exact solution
        formula = nd.FirstDerivativeForward(function, x)
        exact_step, exact_absolute_error = formula.compute_step(
            second_derivative_value, absolute_precision
        )
        print(
            f"Exact step = {exact_step:.4e}, "
            f"exact_absolute_error = {exact_absolute_error:.4e}"
        )
        np.testing.assert_allclose(computed_step, exact_step, rtol=1.0e-5)
        np.testing.assert_allclose(
            computed_absolute_error, exact_absolute_error, rtol=1.0e-5
        )

    def test_first_central(self):
        print("+ test_first_central")
        # Evaluate f'(x) with f(x)= sin(x) using central F.D.
        problem = nd.ScaledExponentialProblem()
        function = problem.get_function()
        function_third_derivative = problem.get_third_derivative()
        x = problem.get_x()
        differentiation_order = 1
        formula_accuracy = 2
        formula = nd.GeneralFiniteDifference(
            function,
            x,
            differentiation_order,
            formula_accuracy,
            direction="central",
        )
        absolute_precision = 1.0e-16
        third_derivative_value = function_third_derivative(x)
        computed_step, computed_absolute_error = formula.compute_step(
            third_derivative_value, absolute_precision
        )
        print(
            f"Computed step = {computed_step:.6e}, "
            f"computed_absolute_error = {computed_absolute_error:.6e}"
        )
        # Check indices
        imin, imax = formula.get_indices_min_max()
        assert imin == -1
        assert imax == 1
        # Check coefficients
        coefficients = formula.get_coefficients()
        np.testing.assert_allclose(2.0 * coefficients, [-1.0, 0.0, 1.0], rtol=1.0e-5)
        # Exact solution
        formula = nd.FirstDerivativeCentral(function, x)
        exact_step, exact_absolute_error = formula.compute_step(
            third_derivative_value, absolute_precision
        )
        print(
            f"Exact step = {exact_step:.6e}, "
            f"exact_absolute_error = {exact_absolute_error:.6e}"
        )
        np.testing.assert_allclose(computed_step, exact_step, rtol=1.0e-5)
        np.testing.assert_allclose(
            computed_absolute_error, exact_absolute_error, rtol=1.0e-5
        )

    def test_second_central_coefficients(self):
        print("+ test_second_central")
        # Evaluate f''(x) with f(x)= sin(x) using central F.D.
        problem = nd.ExponentialProblem()
        function = problem.get_function()
        function_fourth_derivative = problem.get_fourth_derivative()
        x = problem.get_x()
        differentiation_order = 2
        formula_accuracy = 2
        formula = nd.GeneralFiniteDifference(
            function,
            x,
            differentiation_order,
            formula_accuracy,
            direction="central",
        )
        # Check indices
        imin, imax = formula.get_indices_min_max()
        assert imin == -1
        assert imax == 1
        # Check coefficients
        coefficients = formula.get_coefficients()
        np.testing.assert_allclose(2.0 * coefficients, [1.0, -2.0, 1.0], rtol=1.0e-5)
        # Check derivative
        absolute_precision = 1.0e-15
        fourth_derivative_value = function_fourth_derivative(x)
        step, _ = formula.compute_step(fourth_derivative_value, absolute_precision)
        second_derivative_approx = formula.compute(step)
        print(
            f"Computed step = {step:.4e}, "
            f"second_derivative_approx = {second_derivative_approx}"
        )
        # Exact solution
        second_derivative_function = problem.get_second_derivative()
        second_derivative_exact = second_derivative_function(x)
        absolute_error = abs(second_derivative_approx - second_derivative_exact)
        print(
            f"second_derivative_exact = {second_derivative_exact}, "
            f"absolute_error = {absolute_error}"
        )
        np.testing.assert_allclose(
            second_derivative_approx, second_derivative_exact, rtol=1.0e-7
        )
        # Check error
        step = 1.0e-4
        computed_error = formula.compute_error(
            step, fourth_derivative_value, absolute_precision
        )
        exact_error = nd.SecondDerivativeCentral.compute_error(
            step, fourth_derivative_value, absolute_precision
        )
        print(f"computed_error = {computed_error}, exact_error = {exact_error}")
        np.testing.assert_allclose(computed_error, exact_error, rtol=1.0e-7)
        # Check step
        absolute_precision = 1.0e-16
        fourth_derivative_value = function_fourth_derivative(x)
        computed_step, computed_absolute_error = formula.compute_step(
            fourth_derivative_value, absolute_precision
        )
        print(
            f"Computed step = {computed_step:.4e}, "
            f"computed_absolute_error = {computed_absolute_error:.4e}"
        )
        # Exact solution
        formula = nd.SecondDerivativeCentral(function, x)
        exact_step, exact_absolute_error = formula.compute_step(
            fourth_derivative_value, absolute_precision
        )
        print(
            f"Exact step = {exact_step:.4e}, "
            f"exact_absolute_error = {exact_absolute_error:.4e}"
        )
        np.testing.assert_allclose(computed_step, exact_step, rtol=1.0e-5)
        np.testing.assert_allclose(
            computed_absolute_error, exact_absolute_error, rtol=1.0e-5
        )


if __name__ == "__main__":
    unittest.main()
