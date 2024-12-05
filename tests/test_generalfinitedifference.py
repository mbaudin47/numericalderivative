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
        direction = "centered"
        for formula_accuracy in [2, 4, 6]:
            formula = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, direction=direction)
            step = formula.compute_step()
            f_third_derivative_approx = formula.finite_differences(step)
            print(f"formula_accuracy = {formula_accuracy}, "
                  f"step = {step}, "
                  f"f_third_derivative_approx = {f_third_derivative_approx}, ")
            np.testing.assert_almost_equal(f_third_derivative_approx, f_third_derivative_exact, decimal=6)
        # forward and backware formula are ok for even accuracy
        for formula_accuracy in range(3, 5):
            for direction in ["forward", "backward"]:
                formula = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, direction=direction)
                step = formula.compute_step()
                f_third_derivative_approx = formula.finite_differences(step)
                print(f"formula_accuracy = {formula_accuracy}, "
                      f"step = {step}, "
                      f"f_third_derivative_approx = {f_third_derivative_approx}")
                np.testing.assert_almost_equal(f_third_derivative_approx, f_third_derivative_exact, decimal=6)

if __name__ == "__main__":
    unittest.main()
