# numericalderivative

## What is it?

The goal of this project is to compute the first derivative of a function
using finite difference formulas.
The difficulty with these formulas is that it must use a 
step which must be neither too large (otherwise the truncation error dominates 
the error) nor too small (otherwise the condition error dominates).
To solve this issue, the module provides algorithms to compute an approximate
optimal finite difference step.
For this purpose, the module provides exact methods (based on the value 
of higher derivatives) and approximate methods (based on function values).
Furthermore, the module provides finite difference formula for the 
first, second and third derivative of a function.
Furthermore, this package provides 15 benchmark problems for numerical
differentiation.

This module allows you to do this:

```python
import math
import numericalderivative as nd

def scaled_exp(x):
    alpha = 1.0e6
    return math.exp(-x / alpha)


h0 = 1.0e5
x = 1.0e0
algorithm = nd.SteplemanWinarsky(scaled_exp, x)
h_optimal, iterations = algorithm.compute_step(h0)
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
```

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

## Roadmap
- Compute exact step of third derivative from central F.D.
- Structure the GeneralFiniteDifference class to make it consistent with
  other classes.
  Move compute_optimal_step() into FiniteDifferenceOptimalStep or
  move 
  Fix the constant in the optimal step.
  Make compute_indices() and compute_coefficients() as private
  methods: create new accessor methods.
  Initialize these attributes in the constructor.

- Create an example for the DerivativeBenchmarkProblem() class.

- Fix the autosummary of the "Methods" section.
  Notice that it works for DerivativeBenchmarkProblem and fails
  for GillMurraySaundersWright.

- Implement the method of:

Shi, H. J. M., Xie, Y., Xuan, M. Q., & Nocedal, J. (2022). Adaptive finite-difference interval estimation for noisy derivative-free optimization. _SIAM Journal on Scientific Computing_, _44_(4), A2302-A2321.

- Format the mathematical equations in FiniteDifferenceOptimalStep.
