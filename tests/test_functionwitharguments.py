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

def scaled_exp_with_1_arg(x, alpha):
    return np.exp(-x / alpha)

def scaled_exp_with_2_args(x, alpha, beta):
    return beta * np.exp(-x / alpha)

class CheckFunctionWithArguments(unittest.TestCase):
    def test_eval(self):
        x = 1.0
        function = nd.FunctionWithArguments(scaled_exp)
        y = function(x)
        y_exact = scaled_exp(x)
        np.testing.assert_allclose(
            y, y_exact, 
        )
        for i in range(10):
            y = function(x)
        counter = function.get_number_of_evaluations()
        assert counter == 11

    def test_eval_with_1_arg(self):
        x = 1.0
        alpha = 1.e6
        args = [alpha]
        function = nd.FunctionWithArguments(scaled_exp_with_1_arg, args)
        y = function(x)
        y_exact = scaled_exp_with_1_arg(x, alpha)
        np.testing.assert_allclose(
            y, y_exact, 
        )
        for i in range(10):
            y = function(x)
        counter = function.get_number_of_evaluations()
        assert counter == 11

    def test_eval_with_2_args(self):
        x = 1.0
        alpha = 1.e6
        beta = 2.0
        args = [alpha, beta]
        function = nd.FunctionWithArguments(scaled_exp_with_2_args, args)
        y = function(x)
        y_exact = scaled_exp_with_2_args(x, alpha, beta)
        np.testing.assert_allclose(
            y, y_exact, 
        )
        for i in range(10):
            y = function(x)
        counter = function.get_number_of_evaluations()
        assert counter == 11

if __name__ == "__main__":
    unittest.main()
