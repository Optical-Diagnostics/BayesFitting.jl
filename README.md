# FittingUtils

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Optical-Diagnostics.github.io/FittingUtils.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Optical-Diagnostics.github.io/FittingUtils.jl/dev/)
[![Build Status](https://github.com/Optical-Diagnostics/FittingUtils.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Optical-Diagnostics/FittingUtils.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/Optical-Diagnostics/FittingUtils.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/Optical-Diagnostics/FittingUtils.jl)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Parameter fitting with priors: least-squares (`LevenbergMarquardtFit`, via
NonlinearSolve) and Bayesian (`TuringFit`) fits behind one `fit_model`
interface, returning a `FitResult` whose parameters are distributions and which
implements the StatsAPI accessors (`coef`, `stderror`, `vcov`, `confint`,
`predict`, `aic`/`bic`). Also covers multi-dataset fits with shared and
private parameters, and `fit_series` for parameters that drift along an ordered
coordinate.

## Example

```julia
using FittingUtils, NonlinearSolve, ComponentArrays, Distributions

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

## Parameter scaling

Every method works on the parameters in units of their prior width, centred on
the prior mean, so a density of 1e14 and an offset of 1e-3 can be fitted
together. A zero-width prior (`Dirac`) pins its parameter. Give
`scale = (; T = 10.0)` to `fit_model` when a prior is deliberately wide and the
width is not a good unit. Quantities that range over decades are best fitted
as logarithms with a normal prior on the logarithm.
