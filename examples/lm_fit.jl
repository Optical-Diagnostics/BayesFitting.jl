# Run with the examples environment: julia --project=examples examples/lm_fit.jl

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

show(stdout, MIME"text/plain"(), fr); println()
show(stdout, MIME"text/plain"(), mc); println()
