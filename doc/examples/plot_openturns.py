#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 - Michaël Baudin.
"""
Applies Stepleman & Winarsky method to an OpenTURNS function
============================================================
"""

# %%
import openturns as ot
import numericalderivative as nd
from openturns.usecases import chaboche_model
from openturns.usecases import cantilever_beam
from openturns.usecases import fireSatellite_function

# %%
# Chaboche model
# --------------
#
# Load the Chaboche model
cm = chaboche_model.ChabocheModel()
model = ot.Function(cm.model)
inputMean = cm.inputDistribution.getMean()
print(f"inputMean = {inputMean}")
meanStrain, meanR, meanC, meanGamma = inputMean

# %%
# Print the derivative from OpenTURNS
derivative = model.gradient(inputMean)
print(f"derivative = ")
derivative

# %%
# Here is the derivative with default step size in OpenTURNS :
#
# .. code-block::
#
#    derivative = 
#    [[  1.93789e+09 ]
#    [  1           ]
#    [  0.0297619   ]
#    [ -1.33845e+06 ]]

# %%
# Print the gradient function
gradient = model.getGradient()
gradient

# %%
# Derivative with respect to strain
# Define a function which takes only strain as input and returns a float
def functionStrain(strain, r, c, gamma):
    x = [strain, r, c, gamma]
    sigma = model(x)
    return sigma[0]


# %%
# Algorithm to detect h* for Strain
h0 = 1.0e0
args = [meanR, meanC, meanGamma]
algorithm = nd.SteplemanWinarsky(functionStrain, meanStrain, args=args, verbose=True)
h_optimal, iterations = algorithm.compute_step(h0)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print("Optimum h =", h_optimal)
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
print(f"Derivative wrt strain= {f_prime_approx:.17e}")


# %%
# Derivative with respect to R
# Define a function which takes only R as input and returns a float
def functionR(r, strain, c, gamma):
    x = [strain, r, c, gamma]
    sigma = model(x)
    return sigma[0]


# %%
# Algorithm to detect h* for R
h0 = 1.0e9
args = [meanStrain, meanC, meanGamma]
algorithm = nd.SteplemanWinarsky(
    functionR, meanR, args=args, relative_precision=1.0e-14, verbose=True
)
h_optimal, iterations = algorithm.compute_step(h0)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print(f"Optimum h = {h_optimal:e}")
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
print(f"Derivative wrt R= {f_prime_approx:.17e}")


# %%
# Derivative with respect to C
# Define a function which takes only C as input and returns a float
def functionR(c, strain, r, gamma):
    x = [strain, r, c, gamma]
    sigma = model(x)
    return sigma[0]


# %%
# Algorithm to detect h* for C
h0 = 1.0e15
args = [meanStrain, meanR, meanGamma]
algorithm = nd.SteplemanWinarsky(
    functionR, meanC, args=args, relative_precision=1.0e-14, verbose=True
)
h_optimal, iterations = algorithm.compute_step(h0)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print(f"Optimum h = {h_optimal:e}")
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
print(f"Derivative wrt C= {f_prime_approx:.17e}")


# %%
# Derivative with respect to Gamma
# Define a function which takes only C as input and returns a float
def functionGamma(gamma, strain, r, c):
    x = [strain, r, c, gamma]
    sigma = model(x)
    return sigma[0]


# %%
# Algorithm to detect h* for Gamma
h0 = 1.0e0
args = [meanStrain, meanR, meanC]
algorithm = nd.SteplemanWinarsky(
    functionGamma, meanGamma, args=args, relative_precision=1.0e-14, verbose=True
)
h_optimal, iterations = algorithm.compute_step(h0)
number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
print(f"Optimum h = {h_optimal:e}")
print("iterations =", iterations)
print("Function evaluations =", number_of_function_evaluations)
f_prime_approx = algorithm.compute_first_derivative(h_optimal)
print(f"Derivative wrt Gamma= {f_prime_approx:.17e}")


# %%
# Derivative with respect to [strain, r, c, gamma]
def genericFunction(x, xIndex, referenceInput):
    inputDimension = referenceInput.getDimension()
    complementIndices = [i for i in range(inputDimension) if i != xIndex]
    modelInput = ot.Point(inputDimension)
    modelInput[xIndex] = x
    for i in complementIndices:
        modelInput[i] = referenceInput[i]
    y = model(modelInput)
    modelOutput = y[0]
    return modelOutput


# %%
# Default step size for all components
initialStep = ot.Point([1.0e0, 1.0e8, 1.0e8, 1.0e0])
inputDimension = model.getInputDimension()
referenceInput = cm.inputDistribution.getMean()
inputDescription = cm.inputDistribution.getDescription()
optimalStep = ot.Point(inputDimension)
for xIndex in range(inputDimension):
    inputMarginal = referenceInput[xIndex]
    print(
        f"+ Derivative with respect to {inputDescription[xIndex]} "
        f"at point {inputMarginal}"
    )
    args = [xIndex, referenceInput]
    algorithm = nd.SteplemanWinarsky(
        genericFunction,
        inputMarginal,
        args=args,
        relative_precision=1.0e-12,
        verbose=True,
    )
    h_optimal, iterations = algorithm.compute_step(initialStep[xIndex])
    number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
    print(f"    Optimum h = {h_optimal:e}")
    print("    Iterations =", iterations)
    print("    Function evaluations =", number_of_function_evaluations)
    f_prime_approx = algorithm.compute_first_derivative(h_optimal)
    print(f"    Derivative wrt {inputDescription[xIndex]}= {f_prime_approx:.17e}")
    # Store optimal point
    optimalStep[xIndex] = h_optimal

# %%
print("The optimal step for central finite difference is")
print(f"optimalStep = {optimalStep}")

# %%
# Configure the model with the optimal step computed
# from SteplemanWinarsky
gradStep = ot.ConstantStep(optimalStep)  # Constant gradient step
model.setGradient(ot.CenteredFiniteDifferenceGradient(gradStep, model.getEvaluation()))
# Now the gradient uses the optimal step sizes
derivative = model.gradient(inputMean)
print(f"derivative = ")
print(derivative)

# %%
# Derivative with step size computed from SteplemanWinarsky :
#
# .. code-block::
#
#     derivative = 
#     [[  1.93789e+09 ]
#      [  1           ]
#      [  0.0295312   ]  # <- This is a change in the 3d decimal
#      [ -1.33845e+06 ]]

# %%
# Compute the step of a ot.Function using Stepleman & Winarsky
# ------------------------------------------------------------
#
# The function below computes the step of a finite difference formula
# applied to an OpenTURNS function using Stepleman & Winarsky's method.

# %%
def computeSteplemanWinarskyStep(
    model,
    initial_step,
    referenceInput,
    relative_precision=1.0e-16,
    beta=4.0,
    verbose=False,
):
    """
    Uses SteplemanWinarsky to compute a step size for central finite differences

    The central F.D. is:

    f'(x) ~ (f(x + h) - f(x - h)) / (2 * h)

    Parameters
    ----------
    model : ot.Function(inputDimension, 1)
        The model, which output dimension equal to 1.
    initial_step : ot.Point(inputDimension)
        The initial step size.
    referenceInput : ot.Point(inputDimension)
        The point X where the derivative is to be computed.
    relative_precision : float, > 0, optional
        The absolute relative_precision of evaluation of f. The default is 1.0e-16.
    verbose : bool, optional
        Set to True to print intermediate messages. The default is False.

    Returns
    -------
    optimalStep : ot.Point(inputDimension)
        The optimal step for central finite difference.
    """

    def genericFunction(x, xIndex, referenceInput):
        if verbose:
            print("x = ", x)
        inputDimension = referenceInput.getDimension()
        complementIndices = [i for i in range(inputDimension) if i != xIndex]
        modelInput = ot.Point(inputDimension)
        modelInput[xIndex] = x
        for i in complementIndices:
            modelInput[i] = referenceInput[i]
        y = model(modelInput)
        modelOutput = y[0]
        if verbose:
            print("y = ", y)
        return modelOutput

    inputDimension = model.getInputDimension()
    inputDescription = model.getInputDescription()
    optimalStep = ot.Point(inputDimension)
    if verbose:
        print(f"Input dimension = {inputDimension}")
        print(f"Input description = {inputDescription}")
    for xIndex in range(inputDimension):
        inputMarginal = referenceInput[xIndex]
        if verbose:
            print(
                f"+ Derivative with respect to {inputDescription[xIndex]} "
                f"at point {inputMarginal}"
            )
        args = [xIndex, referenceInput]
        algorithm = nd.SteplemanWinarsky(
            genericFunction,
            inputMarginal,
            args=args,
            relative_precision=relative_precision,
            verbose=verbose,
        )
        h_optimal, iterations = algorithm.compute_step(
            initial_step[xIndex],
            beta=beta,
        )
        number_of_function_evaluations = algorithm.get_number_of_function_evaluations()
        f_prime_approx = algorithm.compute_first_derivative(h_optimal)
        if verbose:
            print(f"    Optimum h = {h_optimal:e}")
            print("    Iterations =", iterations)
            print("    Function evaluations =", number_of_function_evaluations)
            print(
                f"    Derivative wrt {inputDescription[xIndex]} = {f_prime_approx:.17e}"
            )
        # Store optimal point
        optimalStep[xIndex] = h_optimal
    return optimalStep


# %%
# Cantilever beam model
# ---------------------
#
# Apply the same method to the cantilever beam model
# Load the cantilever beam model
cb = cantilever_beam.CantileverBeam()
model = ot.Function(cb.model)
inputMean = cb.distribution.getMean()
print(f"inputMean = {inputMean}")

# %%
# Print the derivative from OpenTURNS
derivative = model.gradient(inputMean)
print(f"derivative = ")
print(f"{derivative}")

# %%
# Derivative with OpenTURNS's default step size :
#
# .. code-block::
#
#    derivative = 
#    [[ -2.53725e-12 ]
#     [  0.000567037 ]
#     [  0.200131    ]
#     [ -1.17008e+06 ]]

# %%
# Notice that the CantileverBeam model has an exact gradient in OpenTURNS,
# because it is symbolic.
# Hence, using the optimal step should not make any difference.

# %%
# Compute step from SteplemanWinarsky
initialStep = ot.Point(inputMean) / 2
optimalStep = computeSteplemanWinarskyStep(model, initialStep, inputMean)
print("The optimal step for central finite difference is")
print(f"optimalStep = {optimalStep}")

gradStep = ot.ConstantStep(optimalStep)  # Constant gradient step
model.setGradient(ot.CenteredFiniteDifferenceGradient(gradStep, model.getEvaluation()))
# Now the gradient uses the optimal step sizes
derivative = model.gradient(inputMean)
print(f"derivative = ")
print(f"{derivative}")

# %%
# Derivative with SteplemanWinarskyStep
#
# .. code-block::
#
#     derivative = 
#     [[ -2.53725e-12 ]
#      [  0.000567037 ]
#      [  0.200131    ]
#      [ -1.17008e+06 ]]
#
# We see that this is the correct step size.


# %%
# Fire satellite model
# --------------------
#
# Load the Fire satellite use case with total torque as output
# Print the derivative from OpenTURNS
m = fireSatellite_function.FireSatelliteModel()
inputMean = m.inputDistribution.getMean()
m.modelTotalTorque.setInputDescription(
    ["H", "Pother", "Fs", "theta", "Lsp", "q", "RD", "Lalpha", "Cd"]
)
m.modelTotalTorque.setOutputDescription(["Total torque"])
model = ot.Function(m.modelTotalTorque)
derivative = model.gradient(inputMean)
print(f"derivative = ")
print(f"{derivative}")

# %%
# From OpenTURNS with default settings:
#
# .. code-block::
#
#     derivative = 
#     9x1
#     [[ -4.78066e-10 ]
#      [ -3.46945e-13 ]
#      [  2.30666e-09 ]
#      [  4.90736e-09 ]
#      [  1.61453e-06 ]
#      [  2.15271e-06 ]
#      [  5.17815e-10 ]
#      [  0.00582819  ]
#      [  0.0116564   ]]

# %%
# There is a specific difficulty with FireSatellite model for the derivative 
# with respect to Pother.
# Indeed, the gradient of the TotalTorque with respect to Pother is close 
# to zero.
# Furthermore, the nominal value (mean) of Pother is 1000.
# Therefore, in order to get a sufficiently large number of lost digits, 
# the algorithm is forced to use a very large step h, close to 10^4.
# But this leads to a negative value of Pother - h, which produces
# a math domain error.
# In other words, the model has an input range which is ignored by the 
# algorithm.
# To solve this issue the interval which defines the set of
# possible values of x should be introduced.

# %%
# Compute step from SteplemanWinarsky
initialStep = ot.Point(inputMean) / 2
optimalStep = computeSteplemanWinarskyStep(
    model,
    initialStep,
    inputMean,
    verbose=True,
    relative_precision=1.0e-16,
)
print("The optimal step for central finite difference is")
print(f"optimalStep = {optimalStep}")

# %%
gradStep = ot.ConstantStep(optimalStep)  # Constant gradient step
model.setGradient(ot.CenteredFiniteDifferenceGradient(gradStep, model.getEvaluation()))
# Now the gradient uses the optimal step sizes
derivative = model.gradient(inputMean)
print(f"derivative = ")
print(f"{derivative}")

# %%
# From SteplemanWinarsky
#
# .. code-block::
#
#    derivative = 
#    9x1
#    [[ -4.78157e-10 ]
#     [ -2.91776e-13 ]  # <- This is a minor change
#     [  2.30671e-09 ]
#     [  4.90745e-09 ]
#     [  1.61453e-06 ]
#     [  2.15271e-06 ]
#     [  5.17805e-10 ]
#     [  0.00582819  ]
#     [  0.0116564   ]]

# %%
