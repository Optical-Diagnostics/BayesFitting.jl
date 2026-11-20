abstract type FitMethod end

"""
    TuringFit()

Bayesian fitting by MCMC with Turing.jl: the result's marginals are posterior
histograms. Needs `using Turing`. See [`fit_model`](@ref) for its keywords.
"""
struct TuringFit <: FitMethod end
required_package(::TuringFit) = "Turing"

"""
    LevenbergMarquardtFit()

Least-squares fitting with NonlinearSolve.jl's Levenberg-Marquardt solver: the
result's marginals are `Normal`s from the Laplace approximation at the optimum.
Needs `using NonlinearSolve`. See [`fit_model`](@ref) for its keywords.
"""
struct LevenbergMarquardtFit <: FitMethod end
required_package(::LevenbergMarquardtFit) = "NonlinearSolve"
