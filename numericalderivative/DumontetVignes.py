# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
References
----------
Dumontet, J., & Vignes, J. (1977). 
Détermination du pas optimal dans le calcul des dérivées sur ordinateur. 
RAIRO. Analyse numérique, 11 (1), 13-25.
"""

import numpy as np
from .NumericalDerivative import NumericalDerivative
from .FiniteDifferenceFormula import FiniteDifferenceFormula
from .FiniteDifferenceOptimalStep import FiniteDifferenceOptimalStep


class DumontetVignes(NumericalDerivative):
    def __init__(
        self,
        function,
        x,
        relative_precision=1.0e-14,
        number_of_digits=53,
        ell_1=1.0 / 15.0,
        ell_2=1.0 / 2.0,
        args=None,
        verbose=False,
    ):
        """
        Use Dumontet & Vignes method to compute the optimum step size.

        Uses centered finite difference for f'.
        The method is based on computing the third derivative.
        Then the optimal step for the central formula for the first derivative is computed
        from the third derivative.

        Parameters
        ----------
        function : function
            The function to differentiate.
        x : float
            The point where the derivative is to be evaluated.
        relative_precision : float, > 0, optional
            The relative precision of evaluation of f. The default is 1.0e-16.
        number_of_digits : int
            The maximum number of digits of the floating point system.
        ell_1 : float
            The minimum bound of the L ratio.
        ell_2 : float
            The maximum bound of the L ratio.
        args : list
            A list of optional arguments that the function takes as inputs.
            By default, there is no extra argument and calling sequence of
            the function must be y = function(x).
            If there are extra arguments, then the calling sequence of
            the function must be y = function(x, arg1, arg2, ...) where
            arg1, arg2, ..., are the items in the args list.
        verbose : bool, optional
            Set to True to print intermediate messages. The default is False.

        Returns
        -------
        None.

        """
        self.relative_precision = relative_precision
        self.number_of_digits = number_of_digits
        # Eq. 34, fixed
        self.ell_1 = ell_1
        self.ell_2 = ell_2
        self.ell_3 = 1.0 / ell_2
        self.ell_4 = 1.0 / ell_1
        self.verbose = verbose
        self.finite_difference = FiniteDifferenceFormula(function, x, args)
        super().__init__(function, x, args)

    def compute_ell(self, k):
        """
        Compute the L ratio depending on k.

        Parameters
        ----------
        k : float, > 0
            The finite difference step for the second derivative.

        Returns
        -------
        ell : float
            The ratio f'''sup(x0) / f'''inf(x0).

        """
        t = np.zeros(4)
        t[0] = self.finite_difference.function_eval(self.x + 2 * k)
        t[1] = -self.finite_difference.function_eval(self.x - 2 * k)  # Fixed wrt paper
        t[2] = -2.0 * self.finite_difference.function_eval(self.x + k)
        t[3] = 2.0 * self.finite_difference.function_eval(self.x - k)  # Fixed wrt paper
        a = 0.0
        b = 0.0
        for i in range(4):
            if t[i] > 0.0:
                a += t[i]
            else:
                b += t[i]
        # Eq. 30
        f3inf = (
            a / (1 + self.relative_precision) + b / (1 - self.relative_precision)
        ) / (2 * k**3)
        f3sup = (
            a / (1 - self.relative_precision) + b / (1 + self.relative_precision)
        ) / (2 * k**3)
        if f3inf == 0.0:
            ell = np.inf
            if self.verbose:
                print(f"Warning: f3inf is zero!")
        else:
            ell = f3sup / f3inf
        return ell, f3inf, f3sup

    def compute_third_derivative(
        self,
        iteration_maximum=50,
        kmin=None,
        kmax=None,
        logscale=False,
        markdown=False,
    ):
        """
        Compute an approximate third derivative of the function

        To do this, we must compute an approximately optimal step for the
        third derivative.
        Hence, the main goal is to compute a step k which is supposed to be
        optimal to compute the third derivative f'''(x) using central finite
        differences.
        The finite difference formula for the third derivative is:

        f'''(x) ~ [f(x + 2 * k) - f(x - 2 * k)
                   - 2 * f(x + k) + 2 * f(x - k)] / (2 * k**3)

        The method computes the optimal step k for f'''(x).
        Then this step is used to compute an approximate value of f'''(x).

        Parameters
        ----------
        iteration_maximum : int, optional
            The number of number_of_iterations. The default is 53.
        kmin : float, kmin > 0
            A minimum bound for k. The default is None.
            If no value is provided, the default is to compute the smallest
            possible kmin using number_of_digits and x.
        kmax : float, kmax > kmin > 0
            A maximum bound for k. The default is None.
            If no value is provided, the default is to compute the largest
            possible kmax using number_of_digits and x.
        logscale : bool, optional
            Set to True to use a logarithmic scale when updating
            the step k during the search.
            Set to False to use a linear scale when updating
            the step k during the search.
            The default is False.
        markdown : bool, optional
            If True, then prints a Markdown table of the iterations.

        Returns
        -------
        third_derivative : float
            The approximate value of the third derivative using the step
            k.
        number_of_iterations : int
            The number of number_of_iterations required to reach that optimum.

        """
        if self.verbose:
            print("x = %.3e" % (self.x))
            print(f"iteration_maximum = {iteration_maximum}")

        if markdown:
            print("| Iteration | kmin | kmax | k | f3inf | f3sup | ell |")
            print("|---|---|---|---|---|---|---|")

        if kmin is None:
            kmin = self.x * 2 ** (-self.number_of_digits + 1)  # Eq. 26
        if kmax is None:
            kmax = self.x * 2 ** (self.number_of_digits - 1)
        if self.verbose:
            print("kmin = ", kmin)
            print("kmax = ", kmax)

        # Check kmin and kmax
        ell_kmin, f3inf, f3sup = self.compute_ell(kmin)
        ell_kmax, f3inf, f3sup = self.compute_ell(kmax)
        if self.verbose:
            print("L(kmin) = ", ell_kmin)
            print("L(kmax) = ", ell_kmax)

        if np.isnan(ell_kmax):
            raise ValueError("Cannot evaluate L(kmax). Please update kmax.")

        if ell_kmin == ell_kmax:
            raise ValueError("L(kmin) = L(kmax). Please increase the search range.")

        if ell_kmin > ell_kmax:
            # L is decreasing. The target interval is [L3, L4]
            if ell_kmin < self.ell_3:
                raise ValueError(
                    "L is decreasing and L(kmin) < L3. Please reduce kmin."
                )
            if ell_kmax > self.ell_4:
                raise ValueError(
                    "L is decreasing and L(kmax) > L4. Please increase kmax."
                )
        else:
            # L is increasing. The target interval is [L1, L2]
            if ell_kmin > self.ell_2:
                raise ValueError(
                    "L is increasing and L(kmin) > L2. Please reduce kmin."
                )
            if ell_kmax < self.ell_1:
                raise ValueError(
                    "L is increasing and L(kmax) < L1. Please increase kmax."
                )

        # Search solution using bissection
        k = kmin
        found = False
        for number_of_iterations in range(iteration_maximum):
            if self.verbose:
                print(
                    "+ Iteration = %d, kmin = %.3e, kmax = %.3e"
                    % (number_of_iterations, kmin, kmax)
                )
            if logscale:
                logk = (np.log(kmin) + np.log(kmax)) / 2.0
                k = np.exp(logk)
            else:
                k = (kmin + kmax) / 2.0
            ell, f3inf, f3sup = self.compute_ell(k)
            if self.verbose:
                print(
                    "  k = %.3e, f3inf = %.3e, f3sup = %.3e, ell = %.3e"
                    % (k, f3inf, f3sup, ell)
                )
            if markdown:
                print(
                    "| %d | %.1e | %.1e | %.1e | %.1e | %.1e | %.1e |"
                    % (number_of_iterations, kmin, kmax, k, f3inf, f3sup, ell)
                )
            if ell > self.ell_1 and ell < self.ell_4:
                if ell > self.ell_2 and ell < self.ell_3:
                    if self.verbose:
                        print("  k is too large : reduce kmax")
                    kmax = k
                else:
                    if self.verbose:
                        print("  k is OK : stop")
                    found = True
                    break
            else:
                if self.verbose:
                    print("  k is too small : increase kmin")
                kmin = k
        if found:
            third_derivative = (f3inf + f3sup) / 2.0  # Eq. 27 et 35
        else:
            raise ValueError(
                "Unable to find step after %d number_of_iterations."
                % (number_of_iterations)
            )
        return third_derivative, number_of_iterations

    def compute_step(
        self,
        iteration_maximum=50,
        kmin=None,
        kmax=None,
        logscale=False,
        markdown=False,
    ):
        """
        Compute an approximate optimum step for the first derivative

        This step is approximately optimal for the central finite difference for f'.
        The central finite difference formula for the first derivative is:

        f'(x) ~ [f(x + h) - f(x - h)] / (2 * h)

        The method computes the optimal step k for f'''(x).
        Then this step is used to compute an approximate value of f'''(x).
        This is used to compute the step h for f'.

        Parameters
        ----------
        iteration_maximum : int, optional
            The number of number_of_iterations. The default is 53.
        kmin : float, kmin > 0
            A minimum bound for k. The default is None.
            If no value is provided, the default is to compute the smallest
            possible kmin using number_of_digits and x.
        kmax : float, kmax > kmin > 0
            A maximum bound for k. The default is None.
            If no value is provided, the default is to compute the largest
            possible kmax using number_of_digits and x.
        logscale : bool, optional
            Set to True to use a logarithmic scale when updating
            the step k during the search.
            Set to False to use a linear scale when updating
            the step k during the search.
            The default is False.
        markdown : bool, optional
            If True, then prints a Markdown table of the iterations.

        Returns
        -------
        step : float, > 0
            The finite difference step for the first derivative
        number_of_iterations : int
            The number of iterations used to compute the step

        """
        third_derivative_value, number_of_iterations = self.compute_third_derivative(
            iteration_maximum,
            kmin,
            kmax,
            logscale,
            markdown,
        )
        # Compute the approximate optimal step for the first derivative
        function_value = self.function_eval(self.x)
        absolute_precision = self.relative_precision * abs(function_value)
        fd_optimal_step = FiniteDifferenceOptimalStep(absolute_precision)
        step, _ = fd_optimal_step.compute_step_first_derivative_central(
            third_derivative_value
        )
        return step, number_of_iterations

    def compute_first_derivative(self, step):
        """
        Compute first derivative using central finite difference.

        This is based on the central finite difference formula:

        f'(x) ~ (f(x + h) - f(x - h)) / (2h)

        Parameters
        ----------
        step : float, > 0
            The finite difference step

        Returns
        -------
        f_prime_approx : float
            The approximate first derivative at point x.
        """
        f_prime_approx = self.finite_difference.compute_first_derivative_central(step)
        return f_prime_approx

    def get_number_of_function_evaluations(self):
        """
        Returns the number of function evaluations.

        Returns
        ----------
        number_of_function_evaluations : int
            The number of function evaluations.
        """
        finite_difference_feval = (
            self.finite_difference.get_number_of_function_evaluations()
        )
        total_feval = finite_difference_feval + self.number_of_function_evaluations
        return total_feval
