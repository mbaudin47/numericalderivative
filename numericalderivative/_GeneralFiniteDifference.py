# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Creates a finite difference formula of arbitrary differentiation order or accuracy
"""

import numpy as np
import sys
import numericalderivative as nd
import math
import scipy


class GeneralFiniteDifference:
    """Create a general finite difference formula"""

    def __init__(
        self,
        function,
        x,
        differentiation_order,
        formula_accuracy,
        direction="central",
        args=None,
    ):
        r"""
        Create a general finite difference formula

        Let :math:`d \in \mathbb{N}` be the order of differentation with :math:`d \geq 1` and
        :math:`p \in \mathbb{N}` be the order of precision with :math:`p \geq 1`. 
        The indices are:

        - if the formula is "forward", then :math:`i_\min = 0` and :math:`i_\max = d + p - 1`,
        - if the formula is "backward", then :math:`i_\min = -(d + p - 1)` and :math:`i_\max = 0`,
        - if the formula is "central", then :math:`d + p` must be odd.
          Then :math:`i_\max = \left\lfloor \frac{d + p - 1}{2} \right\rfloor` 
          and :math:`i_\min = -i_\max`.


        Uses the finite difference formula (see (Shi, Xie, Xuan & Nocedal, 2022) eq. 3.2 page 7):

        .. math::

            f^{(d)}(x) = \frac{d!}{h^d} \sum_{i = i_{\min}}^{i_\max} c_i f(x + h i)
                         - \frac{d!}{(d + p)!} b_{d + p} f^{(d + p)}(\xi) h^p

        where :math:`h > 0` is the step, :math:`\boldsymbol{c} \in \mathbb{R}^{d + p}`
        is the vector of coefficients, :math:`i_\min \in \mathbb{N}` is the minimum index,
        :math:`i_\max \in \mathbb{N}` is the maximum index,
        :math:`\xi \in (x, x + h)`
        and :math:`\epsilon_f > 0` is the absolute precision of the
        function evaluation.
        We have :math:`i_\max \geq i_\min`.
        The particular values of :math:`i_\min` and :math:`i_\max` depend on
        the direction of the F.D. formula, the order of differentiation and the
        order of precision see `_compute_indices()`.
        The coefficient :math:`b_{d + p}` is (see (Shi, Xie, Xuan & Nocedal, 2022) eq. page 7):

        .. math::

            b_{d + p} = \sum_{i = i_{\min}}^{i_\max} i^{d + p} c_i.

        The F.D. approximation has order :math:`p`:

        .. math::

            f^{(d)}(x) = \frac{d!}{h^d} \sum_{i = i_{\min}}^{i_\max} c_i f(x + h i)
                         + O(h^p)
        
        when :math:`h \rightarrow 0`.

        If direction is "central", if :math:`p` is odd,
        then the order of precision is actually :math:`p + 1`.
        This implies that any central F.D. formula has an even order of precision.

        The number of coefficients in the system of equations is:

        .. math::
            n_c = i_{max} - i_{min} + 1.
        
        For a "forward" or "backward" finite difference formula,
        the number of unknown is equal to:

        .. math::
            n_c = d + p.

        For a "central" finite difference formula,
        the number of unknown is equal to:

        .. math::
            n_c =
            \begin{cases}
            d + p - 1 & \textrm{ if } d + p \textrm{ is even}, \\
            d + p & \textrm{ if } d + p \textrm{ is odd}.
            \end{cases}

        Let :math:`A \in \mathbb{R}^{n_c \times n_c}`
        be the matrix defined by the equation:

        .. math::

            a_{ji} = i^j
        
        for :math:`i = i_\min, ..., i_\max` and :math:`j = 0, ..., n_c - 1`.
        The matrix :math:`A` is the transpose of a Vandermonde matrix.
        Let :math:`\boldsymbol{b} \in \mathbb{R}^{n_c}` be the vector of
        coefficients defined by the equation:

        .. math::

            b_{j} = 
            \begin{cases}
            0 & \textrm{ if } j \in \{0, ..., d - 1\}, \\
            1 & \textrm{ if } j = d,\\
            0 & \textrm{ if } j \in \{d + 1, ..., n_c - 1\},
            \end{cases}
        
        for :math:`j = 0, ..., n_c - 1`.
        Then the vector of coefficients :math:`\boldsymbol{c} \in \mathbb{R}^{n_c}`
        is the solution of the linear system of equations:

        .. math::

            A \boldsymbol{c} = \boldsymbol{b}.
        

        These coefficiens have some specific properties.
        The sum is zero:

        .. math::
            \sum_{i=i_{min}}^{i_{max}} c_i = 0.

        For a central formula:

        - if :math:`d` is odd, then:

        .. math::
            c_i = -c_{-i}

        for :math:`i=i_{min}, ..., j_{max}` and :math:`c_0 = 0` ;

        - if :math:`d` is even, then:

        .. math::
            c_i = c_{-i}
            
        for :math:`i=i_{min}, ..., i_{max}`.

        For a central formula, if :math:`p` is odd therefore:

        .. math::
            \sum_{i=i_{min}}^{i_{max}} i^{d+p} c_i = 0.

        Parameters
        ----------
        function : function
            The function to differentiate.
        x : float
            The point where the derivative is to be evaluated.
        differentiation_order : int
            The order of the derivative.
            For example differentiation_order = 1 is the first derivative.
        formula_accuracy : int
            The order of precision of the formula.
            For the central F.D. formula, 
            then the formula accuracy is necessarily even.
            If required, increase the formula accuracy by 1 unit.
        direction : str, optional
            The direction of the formula.
            The direction can be "forward", "backward" or "central".
            The default is "central".
        args : list
            A list of optional arguments that the function takes as inputs.
            By default, there is no extra argument and calling sequence of
            the function must be y = function(x).
            If there are extra arguments, then the calling sequence of
            the function must be y = function(x, arg1, arg2, ...) where
            arg1, arg2, ..., are the items in the args list.

        References
        ----------
        - David Eberly. « Derivative approximation by finite differences ». 2008.
        - Nicholas Maxwell. « Notes on the derivation of Finite Difference kernels on regularly spaced grids, using arbitrary sample points ». 2010.
        - Bengt Fornberg. « Classroom Note :Calculation of Weights in Finite Difference Formulas ». In : SIAM Review 40.3 (1998), p. 685-691.
        - B. Fornberg. « Finite difference method ». In : Scholarpedia 6.10 (2011), p. 9685
        - H.Z. Hassan, A.A. Mohamad et G.E. Atteia. « An algorithm for the finite difference approximation of derivatives with arbitrary degree and order of accuracy ». In : Journal of Computational and Applied Mathematics 236.10 (2012), p. 2622-2631. issn : 0377-0427.
        """
        if differentiation_order <= 0:
            raise ValueError(f"Invalid differentiation order {differentiation_order}")
        self.differentiation_order = differentiation_order
        if formula_accuracy <= 0:
            raise ValueError(f"Invalid formula accuracy {formula_accuracy}")
        if (
            direction != "forward"
            and direction != "backward"
            and direction != "central"
        ):
            raise ValueError(f"Invalid direction {direction}.")
        self.direction = direction
        if self.direction == "central" and formula_accuracy % 2 == 1:
            raise ValueError(
                f"Invalid accuracy for a centered formula with even differentiation order: "
                f"direction = {direction} is central, "
                f"formula_accuracy = {formula_accuracy} is odd."
                f" Please increase formula_accuracy by 1."
            )
        self.formula_accuracy = formula_accuracy
        self.function = nd.FunctionWithArguments(function, args)
        self.x = x
        #
        # Setup the formula
        _ = self._compute_indices()
        # Compute the coefficients
        _ = self._compute_coefficients()

    def _compute_indices(self):
        r"""
        Computes the min and max indices for a finite difference formula.

        This function is used by _compute_coefficients() to compute the
        derivative of arbitrary order and arbitrary order of accuracy.

        Parameters
        ----------
        None

        Returns
        -------
        imin : int
            The minimum indice of the f.d. formula.
        imax : int
            The maximum indice of the f.d. formula.
        """
        if self.direction == "forward":
            self.imin = 0
            self.imax = self.differentiation_order + self.formula_accuracy - 1
        elif self.direction == "backward":
            self.imin = -(self.differentiation_order + self.formula_accuracy - 1)
            self.imax = 0
        elif self.direction == "central":
            self.imax = int(
                math.floor((self.differentiation_order + self.formula_accuracy - 1) / 2)
            )
            self.imin = -self.imax
        else:
            raise ValueError(f"Invalid direction {self.direction}")
        return (self.imin, self.imax)

    def get_indices_min_max(self):
        r"""
        Return the indices of the finite difference formula

        Returns
        -------
        imin : int
            The minimum index of the F.D. formula
        imax : int
            The maximum index of the F.D. formula

        Examples
        --------

        >>> import numericalderivative as nd
        >>>
        >>> def scaled_exp(x):
        >>>     alpha = 1.e6
        >>>     return np.exp(-x / alpha)
        >>>
        >>> x = 1.0e-2
        >>> differentiation_order = 3  # Compute f'''
        >>> formula_accuracy = 6  # Use differentiation_order 6 formula
        >>> imin, imax = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy).get_indices_min_max()
        >>> imin, imax = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "forward").get_indices_min_max()
        >>> imin, imax = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "backward").get_indices_min_max()
        >>> imin, imax = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "central").get_indices_min_max()
        """
        return (self.imin, self.imax)

    def get_coefficients(self):
        r"""
        Return the coefficients of the finite difference formula

        Returns
        -------
        coefficients : np.array(number_of_coefficients)
            The coefficients of the F.D. formula

        Examples
        --------
        >>> import numericalderivative as nd
        >>>
        >>> def scaled_exp(x):
        >>>     alpha = 1.e6
        >>>     return np.exp(-x / alpha)
        >>>
        >>> x = 1.0e-2
        >>> differentiation_order = 3  # Compute f'''
        >>> formula_accuracy = 6  # Use differentiation_order 6 formula
        >>> c = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy).get_coefficients()
        >>> c = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "forward").get_coefficients()
        >>> c = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "backward").get_coefficients()
        >>> c = nd.GeneralFiniteDifference(function, x, differentiation_order, formula_accuracy, "central").get_coefficients()
        """
        return self.coefficients

    def _compute_coefficients(self):
        r"""
        Computes the coefficients of the finite difference formula.

        Parameters
        ----------
        None

        Returns
        -------
        c : np.array(differentiation_order + formula_accuracy)
            The coefficicients of the finite difference formula.
        """
        # Compute matrix
        imin, imax = self._compute_indices()
        indices = list(range(imin, imax + 1))
        number_of_coefficients = imax - imin + 1
        A = np.vander(indices, increasing=True).T
        # Compute right-hand side
        b = np.zeros((number_of_coefficients))
        b[self.differentiation_order] = 1.0
        # Solve
        self.coefficients = np.linalg.solve(A, b)
        return self.coefficients

    def compute_error(
        self,
        step,
        higher_order_derivative_value=1.0,
        absolute_precision=sys.float_info.epsilon,
    ):
        r"""
        Computes the total error

        The total error is the sum of the
        rounding error in the finite difference formula and the truncation
        error in the Taylor expansion (see Baudin, 2023) eq. (9.16) page 224 and
        (Shi, Xie, Xuan & Nocedal, 2022) page 7):

        .. math::

            e(h) = \frac{d! \|\boldsymbol{c}\|_1 \epsilon_f}{h^d}
                   + \frac{d!}{(d + p)!} \left|b_{d + p} f^{(d + p)}(x) \right| h^p

        where :math:`h > 0` is the step, :math:`d \in \mathbb{N}` is the
        order of differentation, :math:`p \in \mathbb{N}` is the order
        of precision, :math:`\boldsymbol{c} \in \mathbb{R}^{d + p}` is the vector of weights and
        :math:`\epsilon_f > 0` is the absolute precision of the
        function evaluation and :math:`b_{d + p}` is equal to:

        .. math::

            b_{d + p} = \sum_{i = i_\min}^{i_\max} i^{d + p} c_i.

        In the previous equation, the one-norm of the vector of coefficients is:

        .. math::

            \|\boldsymbol{c}\|_1 = \sum_{i = i_\min}^{i_\max} |c_i|.


        Parameters
        ----------
        step : float, > 0
            The finite difference step.
        higher_order_derivative_value : float
            The value of the derivative of order differentiation_order + formula_accuracy.
            For example, if differentiation_order = 2 and the formula_accuracy = 3, then
            this must be the derivative of order 3 + 2 = 5.
        absolute_precision : float, > 0
            The absolute precision of the function evaluation

        Returns
        -------
        absolute_error : float
            The absolute error.

        References
        ----------
        - M. Baudin (2023). Méthodes numériques. Dunod.
        - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.
        """
        # See (Shi, Xie, Xuan & Nocedal, 2022) eq. (3.3) page 7
        absolute_coefficients = np.sum(np.abs(self.coefficients))
        # Compute b(d + p)
        q = self.differentiation_order + self.formula_accuracy
        constant = 0.0
        for i in range(self.imin, self.imax + 1):
            constant += self.coefficients[i - self.imin] * i**q
        # Compute rounding error
        rounding_error = (
            math.factorial(self.differentiation_order)
            * absolute_coefficients
            * absolute_precision
            / step**self.differentiation_order
        )
        # Compute truncation error
        factor = np.exp(
            scipy.special.gammaln(self.differentiation_order + 1)
            - scipy.special.gammaln(q + 1)
        )
        truncation_error = (
            factor
            * abs(constant * higher_order_derivative_value)
            * step**self.formula_accuracy
        )
        # Compute error
        total_error = rounding_error + truncation_error
        return total_error

    def compute_step(
        self,
        higher_order_derivative_value=1.0,
        absolute_precision=sys.float_info.epsilon,
    ):
        r"""
        Computes the optimal step

        This step minimizes the total error of the derivative
        central finite difference (see `compute()` and `compute_error()`).
        This step minimizes the total error, taking into account for the
        rounding error in the finite difference formula and the truncation
        error in the Taylor expansion (see Baudin, 2023) eq. (9.16) page 224 and
        (Shi, Xie, Xuan & Nocedal, 2022) page 7).

        The optimal step is:

        .. math::

            h^\star = \left(\frac{d}{p} (d + p)! \frac{\|\boldsymbol{c}\|_1}{|b_{d + p}|}
                      \frac{\epsilon_f}{\left|f^{(d + p)}(x)\right|}\right)^{\frac{1}{d + p}}.

        Parameters
        ----------
        higher_order_derivative_value : float
            The value of the derivative of order differentiation_order + formula_accuracy.
            For example, if differentiation_order = 2 and the formula_accuracy = 3, then
            this must be the derivative of order 3 + 2 = 5.
        absolute_precision : float, > 0
            The absolute precision of the function evaluation

        Returns
        -------
        step : float
            The finite difference step.
        absolute_error : float
            The absolute error.

        References
        ----------
        - M. Baudin (2023). Méthodes numériques. Dunod.
        - Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. SIAM Journal on Scientific Computing, 44 (4), A2302-A2321.
        """
        # See (Shi, Xie, Xuan & Nocedal, 2022) eq. (3.3) page 7
        absolute_coefficients = np.sum(np.abs(self.coefficients))
        # Compute b(d + p)
        q = self.differentiation_order + self.formula_accuracy
        constant = 0.0
        for i in range(self.imin, self.imax + 1):
            constant += self.coefficients[i - self.imin] * i**q
        # Compute step
        factor = abs(
            self.differentiation_order
            * math.factorial(q)
            * absolute_coefficients
            / (self.formula_accuracy * constant)
        )
        exponent_argument = abs(
            factor * absolute_precision / higher_order_derivative_value
        )
        step = exponent_argument ** (1.0 / q)
        absolute_error = self.compute_error(
            step,
            higher_order_derivative_value,
            absolute_precision,
        )
        return step, absolute_error

    def compute(self, step):
        r"""
        Computes the degree d approximate derivative of f at point x.

        Parameters
        ----------
        step : float
            The finite difference step.

        Raises
        ------
        ValueError
            If direction is "central", d + formula_accuracy must be odd.

        Returns
        -------
        z : float
            A approximation of the d-th derivative of f at point x.

        Examples
        --------
        The next example computes an approximate third derivative of the sin function.
        To do so, we use a central F.D. formula of order 2.

        >>> import numericalderivative as nd
        >>> import numpy as np
        >>> x = 1.0
        >>> differentiation_order = 3  # Compute f'''
        >>> formula_accuracy = 2  # Use order 2 precision
        >>> formula = nd.GeneralFiniteDifference(
        >>>     np.sin, x, differentiation_order, formula_accuracy
        >>> )
        >>> step = 1.0  # A first guess
        >>> third_derivative = formula.compute(step)

        We can use either forward, backward or central finite differences.

        >>> formula = nd.GeneralFiniteDifference(
        >>>     np.sin,
        >>>     x,
        >>>     differentiation_order,
        >>>     formula_accuracy,
        >>>     "forward"
        >>> )

        We can compute the step provided the derivative of order 5 is known.
        A first guess of this value can be set, or the default value (equal to 1).
        Then the step can be used to compute the derivative.

        >>> formula = nd.GeneralFiniteDifference(
        >>>     np.sin,
        >>>     x,
        >>>     differentiation_order,
        >>>     formula_accuracy,
        >>>     "central"
        >>> )
        >>> step, absolute_error = formula.compute_step()
        >>> third_derivative = formula.compute(step)
        >>> # Set the fifth derivative, if known
        >>> fifth_derivative_value = 1.0  # This may work
        >>> step, absolute_error = formula.compute_step(fifth_derivative_value)
        >>> # Set the absolute error of the function evaluation
        >>> absolute_precision = 1.0e-14
        >>> step, absolute_error = formula.compute_step(
        >>>     fifth_derivative_value, absolute_precision
        >>> )

        Given the step, we can compute the absolute error.

        >>> absolute_error = formula.compute_error(step)
        """
        # Compute the function values
        y = np.zeros((self.differentiation_order + self.formula_accuracy))
        for i in range(self.imin, self.imax + 1):
            y[i - self.imin] = self.function(self.x + i * step)
        # Apply the formula
        z = 0.0
        for i in range(self.imin, self.imax + 1):
            z += self.coefficients[i - self.imin] * y[i - self.imin]
        factor = (
            math.factorial(self.differentiation_order)
            / step**self.differentiation_order
        )
        z *= factor
        return z
