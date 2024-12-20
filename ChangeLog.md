# Change log

## 0.3 release (in-progress)

### Changes

### Added
- New ShiXieXuanNocedalForward.

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
