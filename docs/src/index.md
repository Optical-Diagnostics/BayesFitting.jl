```@meta
CurrentModule = BayesFitting
```

# BayesFitting

Parameter fitting with priors. [`fit_model`](@ref) fits `model(input, θ)` to
data by least squares ([`LevenbergMarquardtFit`](@ref), with
`using NonlinearSolve`) or by MCMC ([`TuringFit`](@ref), with `using Turing`).
Both return a [`FitResult`](@ref): one distribution per parameter, the full
covariance, convergence diagnostics, and the StatsAPI accessors (`coef`,
`stderror`, `vcov`, `confint`, `predict`, `loglikelihood`, `aic`, `bic`).

Priors are a `ComponentVector` of distributions in the layout of `θ`. Every
method works in units of the prior widths ([`ParameterScaling`](@ref)). Data
with per-point uncertainty is a vector of `Normal`s. Plain numbers fit with an
estimated noise level, for least squares only.

Several datasets with shared and private parameters are fitted with
`fit_model([(model1, x1, data1), (model2, x2, data2)], priors, method)`.
[`fit_series`](@ref) fits a series of datasets whose parameters drift along a
coordinate, optionally GP-filtered with [`GaussianDrift`](@ref)
(`using AbstractGPs`).

## Example

```julia
using BayesFitting, NonlinearSolve, ComponentArrays, Distributions

model(x, θ) = @. θ.ampl * exp(-x^2 / (2 * θ.width^2)) + θ.bg
x = range(-3, 3, length = 200)
y = model(x, (ampl = 2.0, width = 0.7, bg = 0.3)) .+ 0.05 .* randn(length(x))
data = Normal.(y, 0.05)

priors = ComponentVector(ampl = Normal(1, 2), width = Normal(1, 0.5), bg = Normal(0, 1))
fr = fit_model(model, x, data, priors, LevenbergMarquardtFit())

coef(fr), stderror(fr), reduced_chisq(fr)
confint(fr).lower.width, confint(fr).upper.width
band = predict(fr; uncertainty = true)

using Turing
mc = fit_model(model, x, data, priors, TuringFit(); init = fr, progress = false)
```

## Reference

```@index
```

```@autodocs
Modules = [BayesFitting]
```
