# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Class to define Dumontet and Vignes algorithm
"""

import numpy as np
import numericalderivative as nd
import math


class DumontetVignes:
    r"""
    Compute an approximately optimal step for the central F.D. formula for the first derivative

    The method is based on computing the third derivative.
    Then the optimal step for the central formula for the first derivative is computed
    from the third derivative.

    The goal of the method is to compute the step of the central finite 
    difference approximation of the first derivative (see 
    (Dumontet & Vignes, 1977) eq. 2 page 13):

    .. math::

        f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}

    where :math:`f` is the function, :math:`x \in \mathbb{R}` is the 
    input point and :math:`h > 0` is the step.
    The optimal step is (see (Dumontet & Vignes, 1977) eq. 24 page 18):

    .. math::

        h^\star 
        = \left( \frac{3 \epsilon_f}{\left|f^{(3)}(x)\right|} \right)^{1/3}

    if :math:`f^{(3)}(x) \neq 0`, where :math:`\epsilon_f > 0` is the absolute 
    error of the function evaluation.
    Notice that (Dumontet & Vignes, 1977) uses the non-classical constant
    1.67 instead of 3 in the previous equation, but this does not have
    a significant impact on the result.
    The goal of the method is to compute :math:`h^\star` using 
    function evaluations only.

    The third derivative is approximated using the central finite difference formula 
    (see (Dumontet & Vignes, 1977) eq. 25 page 18):

    .. math::

        f^{(3)}_k(x) 
        = \frac{f(x + 2k) - f(x - 2k) - 2 (f(x + k) - f(x - k))}{2k^3}

    where :math:`k > 0` is the step used for the third derivative.
    The method introduces :math:`f^{(3)}_{inf}(x)` and :math:`f^{(3)}_{sup}(x)`
    such that:

    .. math::
    
        f^{(3)}_{inf}(x) \leq f^{(3)}_k(x) \leq f^{(3)}_{sup}(x).

    We evaluate the function on 4 points (see (Dumontet & Vignes, 1977) eq. 28 
    page 19, with 2 errors corrected with respect to the original paper):

    .. math::
    
        & T_1 = f(x + 2k), \qquad T_2 = -f(x - 2k), \\
        & T_3 = -2f(x + k) , \qquad T_4 = 2f(x - k).

    Let :math:`A` and :math:`B` defined by (see (Dumontet & Vignes, 1977) page 19):

    .. math::
    
        A = \sum_{i = 1}^4 \max(T_i, 0),  \quad
        B = \sum_{i = 1}^4 \min(T_i, 0).

    The lower and upper bounds of :math:`f^{(3)}(x)` are computed 
    from (see (Dumontet & Vignes, 1977) eq. 30 page 20):

    .. math::
    
        f^{(3)}_{inf}(x)
        = \frac{1}{2 k^3} \left(\frac{A}{1 + \epsilon_r} + \frac{B}{1 - \epsilon_r}\right), \qquad
        f^{(3)}_{sup}(x)
        = \frac{1}{2 k^3} \left(\frac{A}{1 - \epsilon_r} + \frac{B}{1 + \epsilon_r}\right). 

    where :math:`\epsilon_r > 0` is the relative error of the function evaluation.
    We introduce the ratio (see (Dumontet & Vignes, 1977) eq. 32 page 20):

    .. math::
    
        L(k) = \frac{f^{(3)}_{sup}(x)}{f^{(3)}_{inf}(x)}.

    If :math:`0 < f^{(3)}_{inf}(x) < f^{(3)}_{sup}(x)`, then :math:`L(k) > 1`.
    If :math:`f^{(3)}_{inf}(x) < f^{(3)}_{sup}(x) < 0`, then :math:`L(k) < 1`.
    Moreover, if :math:`k \rightarrow 0`, then :math:`L(k) \rightarrow -1`.

    We search for :math:`k` such that the ratio :math:`L` is:

    - neither too close to 1 because it would mean that :math:`k` is too large
      meaning that the truncation error dominates,
    - nor too far away from 1 because it would mean that :math:`k` is too small
      meaning that the rounding error dominates.
    
    Let :math:`k_{inf}` and :math:`k_{sup}` two real numbers representing the 
    minimum and maximum bounds for :math:`k`.
    We search for :math:`k \in [k_{inf}, k_{sup}]` such that (see 
    (Dumontet & Vignes, 1977) eq. 33 page 20):

    .. math::
    
        L(k) \in [L_1, L_2] \cup [L_3, L_4]

    where:

    - :math:`L_3 = 2` and :math:`L_2 = \frac{1}{L_3}`,
    - :math:`L_4 = 15` and :math:`L_1 = \frac{1}{L_4}`.

    This method cannot manage the case where :math:`f(x) = 0`.
    This is because the relative error has no meaning in this case.

    This method does not perform correctly if the third derivative is 
    zero or is close to zero (see (Dumontet & Vignes, 1977) remark page 22).
    This produces a L ratio which is negative, so that there is no value of the 
    step :math:`k` such that the condition is satisfied.
    This happens for example for the :class:`~numericalderivative.PolynomialProblem`,
    which has a zero third derivative for any :math:`x`.

    Parameters
    ----------
    function : function
        The function to differentiate.
    x : float
        The point where the derivative is to be evaluated.
    relative_precision : float, > 0, optional
        The relative precision of evaluation of f.
    absolute_precision : float, > 0, optional
        The absolute precision of evaluation of f.
    number_of_digits : int
        The maximum number of digits of the floating point system.
    ell_3 : float
        The minimum bound of the L ratio.
    ell_4 : float, > ell_1
        The maximum bound of the L ratio.
    args : list
        A list of optional arguments that the function takes as inputs.
        By default, there is no extra argument and calling sequence of
        the function must be y = function(x).
        If there are extra arguments, then the calling sequence of
        the function must be y = function(x, arg1, arg2, ...) where
        arg1, arg2, ..., are the items in the args list.
    verbose : bool, optional
        Set to True to print intermediate messages.

    References
    ----------
    - Dumontet, J., & Vignes, J. (1977). Détermination du pas optimal dans le calcul des dérivées sur ordinateur. RAIRO. Analyse numérique, 11 (1), 13-25.

    Examples
    --------
    Compute the step of a badly scaled function.

    >>> import numericalderivative as nd
    >>>
    >>> def scaled_exp(x):
    >>>     alpha = 1.e6
    >>>     return np.exp(-x / alpha)
    >>>
    >>> x = 1.0e-2
    >>> kmin = 1.0e-10
    >>> kmax = 1.0e+8
    >>> algorithm = nd.DumontetVignes(
    >>>     scaled_exp, x,
    >>> )
    >>> h_optimal, number_of_iterations = algorithm.find_step(kmin=kmin, kmax=kmax)
    >>> f_prime_approx = algorithm.compute_first_derivative(h_optimal)
    """

    def __init__(
        self,
        function,
        x,
        relative_precision=1.0e-15,
        absolute_precision=1.0e-15,
        number_of_digits=53,
        ell_3=2.0,
        ell_4=15.0,
        args=None,
        verbose=False,
    ):
        if relative_precision <= 0.0:
            raise ValueError(
                f"The relative precision must be > 0. "
                f"here relative precision = {relative_precision}"
            )
        self.relative_precision = relative_precision
        if absolute_precision <= 0.0:
            raise ValueError(
                f"The absolute precision must be > 0. "
                f"here absolute precision = {absolute_precision}"
            )
        self.absolute_precision = absolute_precision
        self.number_of_digits = number_of_digits
        if ell_4 <= ell_3:
            raise ValueError(
                f"We must have ell_4 > ell_3, but ell_4 = {ell_4} and ell_3 = {ell_3}"
            )
        # Eq. 34, fixed
        self.ell_3 = ell_3
        self.ell_4 = ell_4
        self.ell_1 = 1.0 / ell_4
        self.ell_2 = 1.0 / ell_3
        self.verbose = verbose
        self.first_derivative_central = nd.FirstDerivativeCentral(function, x, args)
        self.function = nd.FunctionWithArguments(function, args)
        self.x = x
        self.step_history = []

    def get_ell_min_max(self):
        r"""
        Return the minimum and maximum bounds of the L ratio

        The parameters L1 and L2 can be computed from the equations:

        .. math::

            L_2 = \frac{1}{L_3}, \qquad L_1 = \frac{1}{L_4}.

        Returns
        -------
        ell_3 : float, > 0
            The lower bound of the L ratio.
        ell_4 : float, > 0
            The upper bound of the L ratio.
        """
        return [self.ell_3, self.ell_4]

    def compute_ell(self, k):
        r"""
        Compute the L ratio depending on k.

        A negative L ratio indicates that :math:`f'''_{inf}(x)` and :math:`f'''_{sup}(x)`
        do not have the same sign.
        This is because :math:`f'''_{inf}(x) < 0 < f'''_{sup}(x)`.
        This can happen if the exact third derivative is zero.
        In this case, the method cannot perform correctly.

        It may happen that :math:`f'''_{inf}(x)` is zero.
        This prevents from evaluating the L ratio.
        This might happen if the step k is too small (i.e. too close
        to zero).

        Parameters
        ----------
        k : float, > 0
            The finite difference step for the third derivative.

        Returns
        -------
        ell : float
            The ratio :math:`\frac{f'''_{sup}(x)}{f'''_{inf}(x)}`.
        f3inf : float
            The lower bound of the third derivative
        f3sup : float
            The upper bound of the third derivative

        """
        t = np.zeros(4)
        t[0] = self.function(self.x + 2 * k)
        t[1] = -self.function(self.x - 2 * k)  # Fixed wrt paper
        t[2] = -2.0 * self.function(self.x + k)
        t[3] = 2.0 * self.function(self.x - k)  # Fixed wrt paper
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
        if self.verbose:
            if f3sup < f3inf:
                print(f"Warning: f3inf = {f3inf} > f3sup = {f3sup}")
        if f3inf == 0.0:
            ell = np.inf
            if self.verbose:
                print(f"Warning: f3inf is zero!")
        else:
            ell = f3sup / f3inf
        return ell, f3inf, f3sup

    def compute_third_derivative(
        self,
        kmin=None,
        kmax=None,
        iteration_maximum=53,
        logscale=False,
    ):
        r"""
        Compute an approximate third derivative of the function

        To do this, we must compute an approximately optimal step for the
        third derivative.
        Hence, the main goal is to compute a step h which is supposed to be
        optimal to compute the third derivative f'''(x) using central finite
        differences.
        The finite difference formula for the third derivative is:

        .. math::

            f'''(x) \approx \frac{f(x + 2 h) - f(x - 2 h) - 2 f(x + h) + 2 f(x - h)}{2 h^3}

        The method computes the optimal step h for f'''(x).
        Then this step is used to compute an approximate value of f'''(x).

        Parameters
        ----------
        iteration_maximum : int, optional
            The number of number_of_iterations.
        kmin : float, kmin > 0
            A minimum bound for k.
            If no value is provided, the default is to compute the smallest
            possible kmin using number_of_digits and x.
        kmax : float, kmax > kmin > 0
            A maximum bound for k.
            If no value is provided, the default is to compute the largest
            possible kmax using number_of_digits and x.
        logscale : bool, optional
            Set to True to use a logarithmic scale when updating the step k
            during the search. Set to False to use a linear scale when
            updating the step k during the search.

        Returns
        -------
        third_derivative : float
            The approximate value of the third derivative using the step
            k.
        number_of_iterations : int
            The number of number_of_iterations required to reach that optimum.

        """
        if iteration_maximum < 1:
            raise ValueError(
                f"The maximum number of iterations must be > 1, "
                f"but iteration_maximum = {iteration_maximum}"
            )
        fractional_part, _ = math.modf(iteration_maximum)
        if fractional_part != 0.0:
            raise ValueError(
                f"The maximum number of iterations must be an integer, "
                f"but its fractional part is {fractional_part}"
            )
        if self.verbose:
            print("x = %.3e" % (self.x))
            print(f"iteration_maximum = {iteration_maximum}")

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
            raise ValueError(
                f"L(kmin) = L(kmax) with L(kmin) = {ell_kmin} "
                f"and L(kmax) = {ell_kmax}."
                "Please increase the search range."
            )

        if ell_kmin > ell_kmax:
            # L is decreasing. The target interval is [L3, L4]
            if ell_kmin < self.ell_3:
                raise ValueError(
                    f"L is decreasing and L(kmin) = {ell_kmin} < L3 = {self.ell_3}. "
                    "Please reduce kmin."
                )
            if ell_kmax > self.ell_4:
                raise ValueError(
                    f"L is decreasing and L(kmax) = {ell_kmax} > L4 = {self.ell_4}. "
                    "Please increase kmax."
                )
        else:
            # L is increasing. The target interval is [L1, L2]
            if ell_kmin > self.ell_2:
                raise ValueError(
                    f"L is increasing and L(kmin) = {ell_kmin} > L2 = {self.ell_2}. "
                    "Please reduce kmin."
                )
            if ell_kmax < self.ell_1:
                raise ValueError(
                    f"L is increasing and L(kmax) = {ell_kmax} < L1 = {self.ell_1}. "
                    "Please increase kmax."
                )

        # Search solution using bisection
        k = kmin
        found = False
        self.step_history = []
        for number_of_iterations in range(iteration_maximum):
            if self.verbose:
                print(
                    f"+ Iteration = {number_of_iterations}, "
                    f"kmin = {kmin:.3e}, "
                    f"kmax = {kmax:.3e}"
                )
            if logscale:
                logk = (np.log(kmin) + np.log(kmax)) / 2.0
                k = np.exp(logk)
            else:
                k = (kmin + kmax) / 2.0
            self.step_history.append(k)
            ell, f3inf, f3sup = self.compute_ell(k)
            if self.verbose:
                print(
                    f"  k = {k:.3e}, f3inf = {f3inf:.3e}, f3sup = {f3sup:.3e}, ell = {ell:.3e}"
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

    def find_step(
        self,
        kmin=None,
        kmax=None,
        iteration_maximum=53,
        logscale=False,
    ):
        r"""
        Compute an approximate optimum step for the first derivative

        This step is approximately optimal for the central finite difference for f'.
        The central finite difference formula for the first derivative is:

        .. math::

            f'(x) \approx \frac{f(x + h) - f(x - h)}{2 h}

        The method computes the optimal step h for f'''(x).
        Then this step is used to compute an approximate value of f'''(x).
        This is used to compute the step h for f'.

        Parameters
        ----------
        iteration_maximum : int, optional
            The number of number_of_iterations.
        kmin : float, kmin > 0
            A minimum bound for the finite difference step of the third derivative.
            If no value is provided, the default is to compute the smallest
            possible kmin using number_of_digits and x.
        kmax : float, kmax > kmin > 0
            A maximum bound for the finite difference step of the third derivative.
            If no value is provided, the default is to compute the largest
            possible kmax using number_of_digits and x.
        logscale : bool, optional
            Set to True to use a logarithmic scale when updating
            the step k during the search.
            Set to False to use a linear scale when updating
            the step k during the search.

        Returns
        -------
        step : float, > 0
            The finite difference step for the first derivative
        number_of_iterations : int
            The number of iterations used to compute the step

        """
        third_derivative_value, number_of_iterations = self.compute_third_derivative(
            kmin,
            kmax,
            iteration_maximum,
            logscale,
        )
        # Compute the approximate optimal step for the first derivative
        step, _ = nd.FirstDerivativeCentral.compute_step(
            third_derivative_value, self.absolute_precision
        )
        return step, number_of_iterations

    def compute_first_derivative(self, step):
        r"""
        Compute first derivative using central finite difference.

        This is based on the central finite difference formula:

        .. math::

            f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}

        Parameters
        ----------
        step : float, > 0
            The finite difference step

        Returns
        -------
        f_prime_approx : float
            The approximate first derivative at point x.
        """
        f_prime_approx = self.first_derivative_central.compute(step)
        return f_prime_approx

    def get_number_of_function_evaluations(self):
        """
        Returns the number of function evaluations.

        Returns
        -------
        number_of_function_evaluations : int
            The number of function evaluations.
        """
        finite_difference_feval = (
            self.first_derivative_central.get_function().get_number_of_evaluations()
        )
        function_eval = self.function.get_number_of_evaluations()
        total_feval = finite_difference_feval + function_eval
        return total_feval

    def get_step_history(self):
        """
        Return the history of steps during the bissection search.

        Returns
        -------
        step_history : list(float)
            The list of steps k during intermediate iterations of the bissection search.
            This is updated by :meth:`compute_third_derivative`.

        """
        return self.step_history

    def get_relative_precision(self):
        """
        Return the relative precision of the function evaluation

        Returns
        -------
        relative_precision : float
            The relative precision of evaluation of f.

        """
        return self.relative_precision
