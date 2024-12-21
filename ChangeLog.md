# Change log

## 0.3 release (in-progress)

### Added
- New ShiXieXuanNocedalForward.
- New benchmark_method.
- DumontetVignes: new get_ell_min_max.
- DumontetVignes, SteplemanWinarsky, GillMurraySaundersWright:
  new get_step_history.

### Changes
- DerivativeBenchmark: new print and pretty-print.
- DerivativeBenchmark: create any problem from the test point and the
  interval. This can be useful to create a list of test points
  depending on each problem.
- ScaledExponentialProblem: change parametrization.
- DumontetVignes: parametrize depending on ell3 and ell4 instead
  of ell1 and ell2.
- DumontetVignes, SteplemanWinarsky, GillMurraySaundersWright:
  Renamed compute_step into find_step to avoid confusion.
- SteplemanWinarsky: Rename search_step_with_bisection into find_initial_step

### Documentation
- DumontetVignes, SteplemanWinarsky, GillMurraySaundersWright:
  briefly introduce the method and its criteria.

## 0.2 release (2024-12-9)

### Added
- New GeneralFiniteDifference.

### Changes
- Scipy is a dependency
- Renamed DerivativeBenchmark into DerivativeProblem.
  Other problems are renamed *Problem.
- Renamed GillMurraySaundersWrightExponentialDerivativeBenchmark 
  into GMSWExponentialProblem.
- New benchmark problems:  PolynomialProblem, InverseProblem,
  SXXNProblem1, SXXNProblem2, SXXNProblem3,
  SXXNProblem4, OliverProblem1, OliverProblem2, OliverProblem3.
- Removed FiniteDifferenceFormula and FiniteDifferenceOptimalStep.
  These features are equivalently provided by FirstDerivativeForward, 
  FirstDerivativeCentral, SecondDerivativeCentral, ThirdDerivativeCentral.
- Renamed NumericalDerivative into FunctionWithArguments.
- ThirdDerivativeCentral: Implement exact step and error for third derivative
- SteplemanWinarsky.compute_step: separate hmin and hmax into two separate
  input arguments instead of a single list with 2 floats.
  This enables to set each parameter independently.

### Documentation
- Fixed Sphinx API help pages
- Added Examples sections into many docstrings.
- New examples: use benchmark, finite difference formulas.

## 0.1 release (2024-11-30)

- New algorithms: DumontetVignes, SteplemanWinarsky, GillMurraySaundersWright
- New benchmark problems: DerivativeBenchmark, ExponentialDerivativeBenchmark, 
  LogarithmicDerivativeBenchmark, SquareRootDerivativeBenchmark, 
  AtanDerivativeBenchmark, SinDerivativeBenchmark, 
  ScaledExponentialDerivativeBenchmark, 
  GillMurraySaundersWrightExponentialDerivativeBenchmark.
- New benchmark features: BuildBenchmarkList
- New finite difference formulas and optimal steps: 
  FiniteDifferenceFormula, FiniteDifferenceOptimalStep, NumericalDerivative
- New examples
- New Sphinx doc
