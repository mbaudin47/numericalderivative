# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Class to define Gill, Murray, Saunders and Wright algorithm
"""

import numpy as np
import numericalderivative as nd


class GillMurraySaundersWright:
    r"""
    Compute an approximately optimal step for the forward F.D. formula of the first derivative

    The method is based on three steps:

    - compute an approximate optimal step :math:`h_\Phi` for the second
      derivative using central finite difference formula,
    - compute the approximate second derivative using central finite
      difference formula,
    - compute the approximate optimal step for the first derivative using the
      forward finite difference formula.

    Finally, this approximately optimal step can be use to compute the
    first derivative using the forward finite difference formula.

    The goal of the method is to compute the approximation of :math:`f'(x)`
    using the forward finite difference formula (see (G, M, S & W, 1983) eq. 1 page 311):

    .. math::

        f'(x) \approx \frac{f(x + h) - f(x)}{h}

    where :math:`f` is the function, :math:`x \in \mathbb{R}` is the
    input point and :math:`h > 0` is the step.
    If :math:`f''(x) \neq 0`, then the step which minimizes the total error is:

    .. math::

        h^\star = 2 \sqrt{\frac{\epsilon_f}{\left|f''(x)\right|}}

    where :math:`\epsilon_f > 0` is the absolute error of the function evaluation.
    The goal of the method is to compute :math:`h^\star` using
    function evaluations only.
    An approximate value of the second derivative can be computed from the
    central finite difference formula (see (G, M, S & W, 1983) eq. 8 page 314):

    .. math::

        f''(x) \approx \Phi(h_\Phi)
        = \frac{f(x + h_\Phi) - 2 f(x) + f(x - h_\Phi)}{h_\Phi^2}.

    where :math:`\Phi` is the approximation of :math:`f''(x)` from the
    central finite difference formula and :math:`h_\Phi > 0` is the step of
    the second derivative finite difference formula.
    The method is based on the condition error (see (G, M, S & W, 1983) eq. 1 page 315):

    .. math::

        c(h_\Phi) = \frac{4 \epsilon_f}{h_\Phi^2 |\Phi|}.

    The condition error is a decreasing function of :math:`h_\Phi`.
    The algorithm searches for a step :math:`h_\Phi` such that:

    .. math::

        c_{\min} \leq c(h_\Phi) \leq c_{\max}

    where :math:`c_{\min}` and :math:`c_{\max}` are thresholds defined by the
    user.

    This algorithm is a simplified version of the algorithm published in
    (Gill, Murray, Saunders & Wright, 1983) section 3.2 page 316.
    While (Gill, Murray, Saunders & Wright, 1983) simultaneously considers
    the finite difference step of the forward, backward formula for the
    first derivative and the central formula for the second derivative,
    this algorithm only searches for the optimal step for the central
    formula for the second derivative.

    The algorithm can fail in the case where the function is odd
    or approximately linear.
    For example, the function :math:`\sin` is linear at point :math:`x = -\pi`.
    In this case, the second derivative is zero, which produces a
    value of :math:`\Phi` zero or close to zero.
    This produces an infinite value of the condition error.

    In this algorithm fails to produce a consistent step, can compute
    an approximately optimal step using `FirstDerivativeForward.compute_step`.
    Since the value of the second derivative is unknown, we can make the
    hypothesis that :math:`f''(x) \approx 1`.

    Parameters
    ----------
    function : function
        The function to differentiate.
    x : float
        The point where the derivative is approximated.
    relative_precision : float, optional
        The relative error of the function f at the point x.
    c_threshold_min : float, optional, > 0
        The minimum value of the condition error.
    c_threshold_max : float, optional, > c_threshold_min
        The maximum value of the condition error.
    args : list
        A list of optional arguments that the function takes as inputs.
        By default, there is no extra argument and calling sequence of
        the function must be y = function(x).
        If there are extra arguments, then the calling sequence of
        the function must be y = function(x, arg1, arg2, ...) where
        arg1, arg2, ..., are the items in the args list.
    verbose : bool
        Set to True to print intermediate messages

    Returns
    -------
    None.

    References
    ----------
    - Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). Computing forward-difference intervals for numerical optimization. SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.

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
    >>> kmin = 1.0e-8
    >>> kmax = 1.0e8
    >>> algorithm = nd.GillMurraySaundersWright(
    >>>     scaled_exp, x,
    >>> )
    >>> h_optimal, number_of_iterations = algorithm.compute_step(kmin=kmin, kmax=kmax)
    >>> f_prime_approx = algorithm.compute_first_derivative(h_optimal)
    """

    def __init__(
        self,
        function,
        x,
        relative_precision=1.0e-16,
        c_threshold_min=0.001,
        c_threshold_max=0.1,
        args=None,
        verbose=False,
    ):
        if relative_precision <= 0.0:
            raise ValueError(
                f"The relative precision must be > 0. "
                f"here relative precision = {relative_precision}"
            )
        self.relative_precision = relative_precision
        if c_threshold_max <= c_threshold_min:
            raise ValueError(
                f"c_threshold_max = {c_threshold_max} must be greater than "
                f"c_threshold_min = {c_threshold_min}"
            )
        self.c_threshold_min = c_threshold_min
        self.c_threshold_max = c_threshold_max
        self.verbose = verbose
        self.x = x
        self.second_derivative_central = nd.SecondDerivativeCentral(function, x, args)
        self.function = nd.FunctionWithArguments(function, args)
        self.y = self.function(self.x)
        self.absolute_precision = abs(relative_precision * self.y)
        self.first_derivative_forward = nd.FirstDerivativeForward(function, x, args)

    def get_threshold_min_max(self):
        """
        Return the threshold min and max of the condition error

        Returns
        -------
        c_threshold_min : float, > 0
            The minimum value of the threshold of the condition error.
        c_threshold_max : float, > 0
            The maxiimum value of the threshold of the condition error.
        """
        return [self.c_threshold_min, self.c_threshold_max]

    def compute_condition(self, k):
        r"""
        Compute the condition error for given step k.

        This function evaluates the condition error :math:`c(h_\Phi)` of the
        finite difference formula of the second derivative finite difference
        depending on the step :math:`h_\Phi`:

        .. math::

            c(h_\Phi) = \frac{4 \epsilon_f}{h_\Phi^2 |\Phi|}.

        Parameters
        ----------
        k : float
            The step used for the finite difference approximation
            of the second derivative.

        Returns
        -------
        c : float
            The condition error.

        """
        # Eq. 8 page 314
        # We do not use compute_2nd_derivative because y=f(x) is known.
        # This way, we compute it only once.
        phi = (self.function(self.x + k) - 2 * self.y + self.function(self.x - k)) / (
            k**2
        )
        # Eq. 11 page 315
        if phi == 0.0:
            c = np.inf
        else:
            c = 4.0 * self.absolute_precision / (k**2 * abs(phi))
        return c

    def compute_step_for_second_derivative(
        self, kmin, kmax, iteration_maximum=50, logscale=True
    ):
        """
        Compute the optimal step k suitable to approximate the second derivative.

        Then the approximate value of the second derivative can be computed using
        compute_2nd_derivative().

        Parameters
        ----------
        kmin : float, > 0
            The minimum finite difference step k for the second derivative.
        kmax : float, > kmin
            The maximum step finite difference k for the second derivative.
        iteration_maximum : in, optional
            The maximum number of iterations.
        logscale : bool, optional
            Set to True to use a logarithmic scale to update k.
            Set to False to use a linear scale.
            The default is True.

        Returns
        -------
        step_second_derivative : float, > 0
            The optimum step step_second_derivative.
        number_of_iterations : int
            The number of iterations required to compute step_second_derivative.

        """
        if kmin >= kmax:
            raise ValueError(f"kmin = {kmin} must be less than kmax = {kmax}.")
        # Check C(kmin)
        cmin = self.compute_condition(kmin)
        if self.verbose:
            print(f"kmin = {kmin:.3e}, c(kmin) = {cmin:.3e}")
        if cmin >= self.c_threshold_min and cmin <= self.c_threshold_max:
            iteration = 0
            return kmin, iteration
        elif cmin < self.c_threshold_min:
            raise ValueError(
                f"C(kmin) = {cmin} < c_threshold_min = {self.c_threshold_min}. "
                "Please decrease kmin. "
            )
        # Check C(kmax)
        cmax = self.compute_condition(kmax)
        if self.verbose:
            print(f"kmax = {kmax:.3e}, c(kmax) = {cmax:.3e}")
        if cmax >= self.c_threshold_min and cmax <= self.c_threshold_max:
            iteration = 0
            return kmax, iteration
        elif cmax > self.c_threshold_max:
            raise ValueError(
                f"C(kmax) = {cmax} > c_threshold_max = {self.c_threshold_max}. "
                "Please increase kmax. "
            )
        # Now c_threshold_min <= c(kmin) and c_threshold_max < c(kmin)
        # which implies: c_threshold_max < c(kmin)
        # and c(kmax) <= c_threshold_max and c(kmax) < c_threshold_min
        # which implies: c(kmax) < c_threshold_min
        # In summary: c(kmax) < c_threshold_min < c_threshold_max < c(kmin).
        found = False
        for number_of_iterations in range(iteration_maximum):
            if logscale:
                logk = (np.log(kmin) + np.log(kmax)) / 2.0
                step_second_derivative = np.exp(logk)
            else:
                step_second_derivative = (kmin + kmax) / 2.0
            c = self.compute_condition(step_second_derivative)
            if self.verbose:
                print(
                    f"Iter #{number_of_iterations}, "
                    f"kmin = {kmin:.3e}, "
                    f"kmax = {kmax:.3e}, "
                    f"k = {step_second_derivative:.3e}, "
                    f"c(k) = {c:.3e}"
                )
            if c > self.c_threshold_min and c <= self.c_threshold_max:
                if self.verbose:
                    print(
                        f"  c in [{self.c_threshold_min}, {self.c_threshold_max}]: stop!"
                    )
                found = True
                break
            elif c < self.c_threshold_min:
                if self.verbose:
                    print(f"  c(k) < c_threshold_min: reduce kmax.")
                kmax = step_second_derivative
            else:
                if self.verbose:
                    print(f"  c(k) >= c_threshold_min: increase kmin.")
                kmin = step_second_derivative
        if not found:
            raise ValueError(
                f"Unable to find satisfactory step_second_derivative "
                f"after {iteration_maximum} iterations."
                f"The function might be odd or approximately linear. "
                f"Please increase iteration_maximum = {iteration_maximum}."
            )
        return step_second_derivative, number_of_iterations

    def compute_step(self, kmin, kmax, iteration_maximum=50, logscale=True):
        """
        Compute the optimal step suitable to approximate the fist derivative.

        This method computes the approximately optimal step for the second derivative.
        Then the approximate value of the second derivative can be computed using
        compute_2nd_derivative().

        Parameters
        ----------
        kmin : float, > 0
            The minimum step k for the second derivative.
        kmax : float, > kmin
            The maximum step k for the second derivative.
        iteration_maximum : in, optional
            The maximum number of iterations. The default is 50.
        logscale : bool, optional
            Set to True to use a logarithmic scale to update k.
            Set to False to use a linear scale.
            The default is True.

        Returns
        -------
        step : float, > 0
            The optimum step for the first derivative.
        number_of_iterations : int
            The number of iterations required to compute the step.

        """
        step_second_derivative, number_of_iterations = (
            self.compute_step_for_second_derivative(
                kmin, kmax, iteration_maximum, logscale
            )
        )
        # Compute an approximate 2nd derivative from the approximately optimal step
        second_derivative_value = self.second_derivative_central.compute(
            step_second_derivative
        )
        # Plug the step for second derivative, evaluate the second derivative,
        # and plug it into the formula.
        step, _ = nd.FirstDerivativeForward.compute_step(
            second_derivative_value, self.absolute_precision
        )
        return step, number_of_iterations

    def compute_first_derivative(self, step):
        r"""
        Compute an approximate first derivative using finite differences

        This method uses the formula:

        .. math::

            f'(x) \approx \frac{f(x + h) - f(x)}{h}

        Parameters
        ----------
        step : float, > 0
            The step size.

        Returns
        -------
        f_prime_approx : float
            The approximation of f'(x).
        """
        f_prime_approx = self.first_derivative_forward.compute(step)
        return f_prime_approx

    def get_number_of_function_evaluations(self):
        """
        Returns the number of function evaluations.

        Returns
        -------
        number_of_function_evaluations : int
            The number of function evaluations.
        """
        second_derivative_central_feval = (
            self.second_derivative_central.get_function().get_number_of_evaluations()
        )
        first_derivative_forward = (
            self.first_derivative_forward.get_function().get_number_of_evaluations()
        )
        function_eval = self.function.get_number_of_evaluations()
        total_feval = (
            second_derivative_central_feval + first_derivative_forward + function_eval
        )
        return total_feval
