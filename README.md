# numericalderivative

## What is it?

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

This module makes it possible to do this:

```python
import math
import numericalderivative as nd

def scaled_exp(x):
    alpha = 1.0e6
    return math.exp(-x / alpha)


h0 = 1.0e5  # This is the initial step size
x = 1.0e0
algorithm = nd.SteplemanWinarsky(scaled_exp, x)
h_optimal, iterations = algorithm.find_step(h0)
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
```

## Why is it useful?

Compared to other Python packages, this module has some advantages.
- If the order of magnitude of some higher derivative is known,
  the features provides methods to compute an approximately optimal
  differentiation step.
- If only function values are available, the package provides algorithms
  to compute an approximately optimal step.
  Indeed, other packages provides finite difference formulas, including
  e.g. [numpy.gradient](https://numpy.org/doc/stable/reference/generated/numpy.gradient.html),
  [GitHub - findiff](https://github.com/maroba/findiff),
  [Scipy Docs - differentiate](http://scipy.github.io/devdocs/reference/generated/scipy.differentiate.derivative.html) and
  [GitHub - numdifftools](https://github.com/pbrod/numdifftools).
  But these tools require the user to provide the differentiation
  step, which may be unknown.
  The goal of the current package is to compute this step.

## Documentation & references

- [Package documentation](https://mbaudin47.github.io/numericalderivative/main/index.html)

## Authors

* Michaël Baudin, 2024

## Installation

To install from Github:

```bash
git clone https://github.com/mbaudin47/numerical_derivative.git
cd numerical_derivative
python setup.py install
```

To install from Pip:

```bash
pip install numericalderivative
```

## References
- Gill, P. E., Murray, W., Saunders, M. A., & Wright, M. H. (1983). 
  Computing forward-difference intervals for numerical optimization. 
  SIAM Journal on Scientific and Statistical Computing, 4(2), 310-321.
- Adaptive numerical differentiation
  R. S. Stepleman and N. D. Winarsky
  Journal: Math. Comp. 33 (1979), 1257-1264 
- Dumontet, J., & Vignes, J. (1977). 
  Détermination du pas optimal dans le calcul des dérivées sur ordinateur. 
  RAIRO. Analyse numérique, 11 (1), 13-25.
- Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). 
  Adaptive finite-difference interval estimation for noisy 
  derivative-free optimization. _SIAM Journal on Scientific Computing_, 
  _44_(4), A2302-A2321.

## Roadmap
- Implement a method to compute the absolute error of evaluation of the function f, 
  for example :

    J. J. Moré and S. M. Wild, _Estimating computational noise_, 
    _SIAM Journal on Scientific Computing_, 33 (2011), pp. 1292–1314.

- Make sure that the API help page of each method has a paragraph on the
  cases of failure.
  Also, add a paragraph on an alternative method using compute_step of F.D.
  formula, with some extra assumption on a higher derivative.
