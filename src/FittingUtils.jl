module FittingUtils

using Statistics
import StatsAPI
import LinearAlgebra
using ComponentArrays

import ForwardDiff, FiniteDiff
import Distributions
using EmpiricalDistributions

include("method.jl")
export TuringFit, LevenbergMarquardtFit

include("scaling.jl")
export ParameterScaling

include("datasets.jl")

include("results.jl")
export FitResult
using StatsAPI: coef, coefnames, stderror, nobs, dof, dof_residual, rss, weights,
    residuals, vcov, confint, predict, loglikelihood, aic, bic
using Statistics: cor
export coef, coefnames, stderror, nobs, dof, dof_residual, rss, weights, residuals,
    mse, reduced_chisq, vcov, cor, converged, confint, predict, loglikelihood, aic, bic

include("fit_model.jl")
export fit_model, inverse_sigmas

include("series.jl")
export fit_series, trajectory, GaussianDrift, StaticDrift, Tracked, NotTracked,
    DriftModel, SeriesFitResult


end
