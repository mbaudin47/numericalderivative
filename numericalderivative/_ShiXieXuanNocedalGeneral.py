# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Class to define Shi, Xie, Xuan & Nocedal algorithm for the general finite difference formula
"""

import numpy as np
import numericalderivative as nd
import math


class ShiXieXuanNocedalGeneral:
    r"""
    Compute an approximately optimal step for the forward F.D. formula of any derivative

    Uses general finite difference formula to compute an approximate value of :math:`f^{(d)}(x)`:

    .. math::

        f^{(d)}(x) \approx \frac{d!}{h^d} \sum_{i = i_{\min}}^{i_\max} c_i f(x + h i)

    where :math:`f` is the function, :math:`x \in \mathbb{R}` is the point,
    :math:`h > 0` is the differentiation step, :math:`d \in \mathbb{N}` is the
    differentiation order and :math:`(c_i)_{i_\min \leq i\leq i_\max}` are
    the weights.
    The weights are computed so that the formula has order :math:`p\geq 1`:
    see :class:`~numericalderivative.GeneralFiniteDifference` for details on this
    topic.
    If :math:`f^{(d + p)}(x) \neq 0`, then the step which minimizes the total error is
    (see (Shi, Xie, Xuan & Nocedal, 2022) eq. 3.3 page 9):

    .. math::

        h^\star = \left(\frac{d}{p} (d + p)! \frac{\|\boldsymbol{c}\|_1}{|b_{d + p}|}
        \frac{\epsilon_f}{\left|f^{(d + p)}(x)\right|}\right)^{\frac{1}{d + p}}

    where :math:`\epsilon_f > 0` is the absolute error of the function evaluation
    and :math:`p \in \mathbb{N}` is the order of precision of the formula.
    If the order of magnitude of the order :math:`d + p` derivative can be guessed,
    then :meth:`~numericalderivative.GeneralFiniteDifference.compute_step` can be used.
    Alternatively, the goal of :class:`~numericalderivative.ShiXieXuanNocedalGeneral`
    is to compute :math:`h^\star` using
    function evaluations only and without estimating :math:`f^{(d + p)}(x)`.

    The algorithm considers the test ratio
    (see (Shi, Xie, Xuan & Nocedal, 2022) eq. 3.14 page 7, with correction):

    .. math::

        r(h) = \frac{\left|\left(\widetilde{f}^{(d)}(x; h) -
            \widetilde{f}^{(d)}(x; \alpha h) \right) h^d\right|}{A \epsilon_f}.

    where :math:`h > 0` is the step and :math:`\epsilon_f> 0` is the absolute precision of evaluation
    of the function.
    The goal of the algorithm is to find the step such that
    (see (Shi, Xie, Xuan & Nocedal, 2022) eq. 2.4 page 4):

    .. math::

        r_\ell \leq r(h) \leq r_u

    where :math:`r_\ell > 1` is the lower bound of the test ratio
    and :math:`r_u > r_\ell + 2` is the upper bound.
    The algorithm is based on bisection.

    If the algorithm succeeds, the method produces a step
    :math:`\widetilde{h}` such that:

    .. math::

        \widetilde{h} \in \left[(r_\ell - 1)^{\frac{1}{d + p}},
            (r_u + 1)^{\frac{1}{d + p}}\right]
            \left(\frac{\epsilon_f}{\left|\overline{b}_{d + p} f^{(d + p)}(x)\right|}\right)^{\frac{1}{d + p}}

    where:

    .. math::

        \overline{b}_{d + p} = \frac{d! (1 - \alpha^p)  b_{d + p}}{(d + p)!}.

    Parameters
    ----------
    function : :class:`~numericalderivative.GeneralFiniteDifference`
        The function to differentiate.
    x : float
        The point where the derivative is to be evaluated.
    differentiation_order : int
        The order of differentiation.
            For example differentiation_order = 1 is the first derivative.
    formula_accuracy : int
        The order of precision of the formula.
        For the central F.D. formula,
        then the formula accuracy is necessarily even.
        If required, increase the formula accuracy by 1 unit.
    absolute_precision : float, > 0, optional
        The absolute precision of evaluation of f.
        If the function value is close to zero (e.g. for the sin function
        at x = np.pi where f(x) is close to 1.0e-32), then the absolute
        precision cannot always be computed from the relative precision.
    minimum_test_ratio : float, > 1
        The minimum value of the test ratio.
    maximum_test_ratio : float, > minimum_test_ratio
        The maximum value of the test ratio.
    alpha_parameter : float, > 0, != 1
        The parameter alpha used in the test ratio
    step_factor : float, > 1
        The multiplier of the step
        This is used to update the step in the search algorithm
    args : list
        A list of optional arguments that the function takes as inputs.
        By default, there is no extra argument and calling sequence of
        the function must be y = function(x).
        If there are extra arguments, then the calling sequence of
        the function must be y = function(x, arg1, arg2, ...) where
        arg1, arg2, ..., are the items in the args list.
    verbose : bool, optional
        Set to True to print intermediate messages.

    Returns
    -------
    None.

    References
    ----------
    - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.

    See also
    --------
    FirstDerivativeForward, GeneralFiniteDifference

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
    >>> algorithm = nd.ShiXieXuanNocedalGeneral(
    >>>     scaled_exp, x,
    >>> )
    >>> h_optimal, number_of_iterations = algorithm.find_step()
    >>> f_prime_approx = algorithm.compute_first_derivative(h_optimal)

    Set the initial step.

    >>> initial_step = 1.0e8
    >>> h_optimal, number_of_iterations = algorithm.find_step(initial_step)
    """

    def __init__(
        self,
        general_finite_difference,
        absolute_precision=1.0e-15,
        minimum_test_ratio=1.5,
        maximum_test_ratio=6.0,
        alpha_parameter=2.0,
        step_factor=4.0,
        verbose=False,
    ):
        if not isinstance(general_finite_difference, nd.GeneralFiniteDifference):
            raise ValueError(
                f"The general_finite_difference is not a GeneralFiniteDifference: "
                f"type = {type(general_finite_difference)}"
            )
        self.general_finite_difference = general_finite_difference
        if absolute_precision <= 0.0:
            raise ValueError(
                f"The absolute precision must be > 0. "
                f"here absolute precision = {absolute_precision}"
            )
        self.absolute_precision = absolute_precision
        self.verbose = verbose
        self.step_history = []
        if minimum_test_ratio <= 1.0:
            raise ValueError(
                f"The minimum test ratio must be > 1, "
                f"but minimum_test_ratio = {minimum_test_ratio}"
            )
        if maximum_test_ratio <= minimum_test_ratio:
            raise ValueError(
                f"The maximum test ratio must be greater than the minimum, "
                f"but minimum_test_ratio = {minimum_test_ratio} "
                f" and maximum_test_ratio = {maximum_test_ratio}"
            )
        self.minimum_test_ratio = minimum_test_ratio
        self.maximum_test_ratio = maximum_test_ratio
        if alpha_parameter <= 0.0 or alpha_parameter == 1.0:
            raise ValueError(
                f"alpha must be > 0 and different from 1, "
                f"but alpha_parameter = {alpha_parameter}"
            )
        self.alpha_parameter = alpha_parameter
        if step_factor < 1.0:
            raise ValueError(
                f"step factor must be > 1, " f"but step_factor = {step_factor}"
            )
        self.step_factor = step_factor
        # Compute the A parameter
        # This parameter is so that the scaled weights have a 1-norm
        # equal to one.
        differentiation_order = general_finite_difference.get_differentiation_order()
        coefficients = general_finite_difference.get_coefficients()
        coefficients_1_norm = np.linalg.norm(coefficients, 1)
        self.a_parameter = math.factorial(differentiation_order) * coefficients_1_norm
        return

    def get_ratio_min_max(self):
        r"""
        Return the minimum and maximum of the test ratio

        Returns
        -------
        minimum_test_ratio : float, > 0
            The lower bound of the test ratio.
        maximum_test_ratio : float, > 0
            The upper bound of the test ratio.
        """
        return [self.minimum_test_ratio, self.maximum_test_ratio]

    def compute_test_ratio(self, step):
        r"""
        Compute the test ratio

        Parameters
        ----------
        step : float, > 0
            The finite difference step
        alpha_parameter : float
            The alpha parameter

        Returns
        -------
        test_ratio : float, > 0
            The test ratio
        """
        derivative_approx = self.general_finite_difference.compute(step)
        derivative_approx_alpha = self.general_finite_difference.compute(
            self.alpha_parameter * step
        )
        differentiation_order = (
            self.general_finite_difference.get_differentiation_order()
        )
        test_ratio = (
            abs(derivative_approx - derivative_approx_alpha)
            * step**differentiation_order
            / (self.a_parameter * self.absolute_precision)
        )
        return test_ratio

    def find_step(
        self,
        initial_step=None,
        iteration_maximum=53,
        logscale=True,
    ):
        r"""
        Compute an approximate optimum step for central derivative using monotony properties.

        This function computes an approximate optimal step h for the central
        finite difference.

        If it is not provided by the user, the default initial step is based
        on the hypothesis that the higher order derivative :math:`f^{(d + p)}(x)`
        is equal to 1 and is computed from :meth:`~numericalderivative.GeneralFiniteDifference.compute_step`.
        This initial guess is not always accurate and can lead to failure
        of the algorithm.

        Parameters
        ----------
        initial_step : float, > 0
            The initial step in the algorithm.
        iteration_maximum : int, optional
            The number of number_of_iterations.
        logscale : bool, optional
            Set to True to use a logarithmic scale when updating the step k
            during the search. Set to False to use a linear scale when
            updating the step k during the search.

        Returns
        -------
        estim_step : float
            A step size which is near to optimal.
        number_of_iterations : int
            The number of iterations required to reach that optimum.

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

        if initial_step is None:
            higher_order_derivative_value = 1.0
            estim_step = self.general_finite_difference.compute_step(
                higher_order_derivative_value, self.absolute_precision
            )
        if initial_step < 0.0:
            raise ValueError(
                f"The initial step must be > 0, "
                f"but initial_step = {initial_step:.3e}"
            )
        estim_step = initial_step
        if self.verbose:
            x = self.general_finite_difference.get_x()
            print(f"x = {x}")
            print(f"absolute_precision = {self.absolute_precision:.3e}")
            print(f"estim_step={estim_step:.3e}")
        lower_bound = 0.0
        upper_bound = np.inf
        self.step_history = []
        found = False
        for number_of_iterations in range(iteration_maximum):
            """
            # Check that the upper bound of the step is not too small
            # This would prevent the magic trick to be used, indicating
            # that the problem is inconsistent with the method.
            actual_step = (self.x + estim_step) - self.x  # Magic trick
            if actual_step == 0.0:
                raise ValueError(f"The actual step is zero at x = {self.x}. "
                                 f"The method cannot be used for this problem.")
            """
            # Update history
            self.step_history.append(estim_step)
            test_ratio = self.compute_test_ratio(estim_step)
            if self.verbose:
                print(
                    f"+ Iter.={number_of_iterations}, "
                    f"lower_bound={lower_bound:.3e}, "
                    f"upper_bound={upper_bound:.3e}, "
                    f"estim_step={estim_step:.3e}, "
                    f"r = {test_ratio:.3e}"
                )
            if test_ratio < self.minimum_test_ratio:
                if self.verbose:
                    print(
                        "    - test_ratio < self.minimum_test_ratio. "
                        "Set lower bound to h."
                    )
                lower_bound = estim_step
            elif test_ratio > self.maximum_test_ratio:
                if self.verbose:
                    print(
                        "    - test_ratio > self.minimum_test_ratio. "
                        "Set upper bound to h."
                    )
                upper_bound = estim_step
            else:
                if self.verbose:
                    print(f"    - Step = {estim_step} is OK: stop.")
                found = True
                break
            if upper_bound == np.inf:
                if self.verbose:
                    print("    - upper_bound == np.inf: increase h.")
                estim_step *= self.step_factor
            elif lower_bound == 0.0:
                if self.verbose:
                    print("    - lower_bound == 0: decrease h.")
                estim_step /= self.step_factor
            else:
                if logscale:
                    log_step = (np.log(lower_bound) + np.log(upper_bound)) / 2.0
                    estim_step = np.exp(log_step)
                else:
                    estim_step = (lower_bound + upper_bound) / 2.0
                if self.verbose:
                    print(f"    - Bisection: estim_step = {estim_step:.3e}.")

        if not found:
            raise ValueError(
                f"Unable to find satisfactory step_second_derivative "
                f"after {iteration_maximum} iterations. "
                f"Please increase iteration_maximum = {iteration_maximum}."
            )
        return estim_step, number_of_iterations

    def compute_derivative(self, step):
        """
        Compute an approximate value of f'(x) using central finite difference.

        Parameters
        ----------
        step : float, > 0
            The step size.

        Returns
        -------
        derivative_approx : float
            The approximation of the d-th derivative of f at point x.
        """
        derivative_approx = self.general_finite_difference.compute(step)
        return derivative_approx

    def get_number_of_function_evaluations(self):
        """
        Returns the number of function evaluations.

        Returns
        -------
        number_of_function_evaluations : int
            The number of function evaluations.
        """
        total_feval = (
            self.general_finite_difference.get_function().get_number_of_evaluations()
        )
        return total_feval

    def get_step_history(self):
        """
        Return the history of steps during the bissection search.

        Returns
        -------
        step_history : list(float)
            The list of steps k during intermediate iterations of the bissection search.
            This is updated by :meth:`find_step`.

        """
        return self.step_history

    def get_absolute_precision(self):
        """
        Return the absolute precision of the function evaluation

        Returns
        -------
        absolute_precision : float
            The absolute precision of evaluation of f.

        """
        return self.absolute_precision

    def get_a_parameter(self):
        """
        Return the A parameter

        Returns
        -------
        a_parameter : float
            The A parameter

        """
        return self.a_parameter

    def get_step_factor(self):
        """
        Return the step multiplier

        This is used to update the step at each step of the search algorithm

        Returns
        -------
        step_factor : float
            The multiplier of the step

        """
        return self.step_factor

    def get_alpha_parameter(self):
        """
        Return the step multiplier

        This is used to update the step at each step of the search algorithm

        Returns
        -------
        alpha_parameter : float
            The parameter involved in the test ratio

        """
        return self.alpha_parameter
