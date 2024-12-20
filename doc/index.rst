.. numericalderivative documentation master file, created by
   sphinx-quickstart on Mon Feb 14 09:17:40 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

numericalderivative documentation
=================================

.. image:: _static/error_vs_h.png
     :align: left
     :scale: 50%

numericalderivative is a module for numerical differentiation.

The goal of this project is to compute the derivative of a function
using finite difference formulas.
The difficulty with these formulas is that it must use a 
step which must be neither too large (otherwise the truncation error dominates 
the error) nor too small (otherwise the condition error dominates).
For this purpose, it provides exact methods (based on the value 
of higher derivatives) and approximate methods (based on function values).
Furthermore, the module provides finite difference formulas for the 
first, second, third or any arbitrary order derivative of a function.
Finally, this package provides 15 benchmark problems for numerical
differentiation.

Documentation about numericalderivative can be found `here <https://mbaudin47.github.io/numericalderivative/main/index.html>`_

User documentation
------------------

.. toctree::
   :maxdepth: 3

   user_manual/user_manual

Examples 
--------

.. toctree::
   :maxdepth: 3

   examples/examples

References
----------
- Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). Computing forward-difference intervals for numerical optimization. SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
- Adaptive numerical differentiation. R. S. Stepleman and N. D. Winarsky. Journal: Math. Comp. 33 (1979), 1257-1264
- Dumontet, J., & Vignes, J. (1977). Détermination du pas optimal dans le calcul des dérivées sur ordinateur. RAIRO. Analyse numérique, 11 (1), 13-25.

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
