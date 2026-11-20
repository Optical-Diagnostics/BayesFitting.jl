"""
    FitResult(model, input, marginals, residuals, weights)

Result of a fit.

- `marginals`: one distribution per parameter (a `Normal` from the Laplace
  approximation for `LevenbergMarquardtFit`, a posterior marginal for
  `TuringFit`). Components may themselves be vectors of distributions.
- `residuals`: data residuals only, without prior terms, so
  `nobs`/`dof_residual`/`rss` count observations.
- `weights`: per-point weights `1 / σᵢ` when the data carried its own uncertainty,
  `nothing` when it did not.
- `vcov`: full parameter covariance in physical units, ordered like the
  flattened parameter vector (`getdata(coef(fr))`). `nothing` when the fitting
  method did not provide one.
- `diagnostics`: NamedTuple of convergence diagnostics, `nothing` when the method
  provides none. Always carries `converged::Bool`; `LevenbergMarquardtFit`
  adds `retcode`, `covariance_ok` and `at_bound` (labels of the parameters at
  a `Uniform` bound), `TuringFit` adds `rhat`.
"""
struct FitResult{F, X, P, R, W, C, S}
    model::F
    input::X
    marginals::P
    residuals::R
    weights::W
    vcov::C
    diagnostics::S
end

FitResult(model, input, marginals, residuals, weights) =
    FitResult(model, input, marginals, residuals, weights, nothing, nothing)
FitResult(model, input, marginals, residuals, weights, vcov) =
    FitResult(model, input, marginals, residuals, weights, vcov, nothing)

"""
    converged(fr) -> Bool

Whether the fit converged: solver success and a usable covariance for
`LevenbergMarquardtFit`, all split-R̂ below threshold for `TuringFit`.
Details are in `fr.diagnostics`. Errors when the result carries no convergence
information.
"""
function converged(fr::FitResult)
    fr.diagnostics === nothing && error(
        "This FitResult carries no convergence information (fr.diagnostics === nothing).")
    return fr.diagnostics.converged
end

Base.getindex(fr::FitResult, k::Symbol) = fr.marginals[k]

function init_point(init::FitResult, priors)
    c = StatsAPI.coef(init)
    getaxes(c) == getaxes(priors) || error(
        "`init` FitResult must have the same component structure as the priors " *
        "(getaxes(coef(init)) != getaxes(priors)).")
    return c
end
function init_point(init, priors)
    getaxes(init) == getaxes(priors) || error(
        "`init` must have the same component structure as the priors " *
        "(getaxes(init) != getaxes(priors)).")
    return map(Statistics.mean, init)
end

_component_apply(func, v::AbstractArray) = map(func, v)
_component_apply(func, v) = func(v)

_component_func(ps, func) =
    ComponentArrays.ComponentVector(; [k => _component_apply(func, ps[k]) for k in keys(ps)]...)


StatsAPI.coef(fr::FitResult) = _component_func(fr.marginals, Statistics.mean)
StatsAPI.coefnames(fr::FitResult) = keys(fr.marginals)
StatsAPI.stderror(fr::FitResult) = _component_func(fr.marginals, Statistics.std)

"""
    vcov(fr) -> Matrix

Full parameter covariance in physical units, ordered like the flattened
parameter vector (`getdata(coef(fr))`).
"""
function StatsAPI.vcov(fr::FitResult)
    fr.vcov === nothing && error(
        "This FitResult carries no covariance matrix (fr.vcov === nothing).")
    return fr.vcov
end

"""
    cor(fr) -> Matrix

Parameter correlation matrix from [`vcov`](@ref). Rows of a pinned parameter
(zero variance) are zero, including the diagonal.
"""
function Statistics.cor(fr::FitResult)
    Σ = StatsAPI.vcov(fr)
    s = sqrt.(LinearAlgebra.diag(Σ))
    s⁻ = map(x -> x > 0 ? inv(x) : zero(x), s)
    C = LinearAlgebra.Diagonal(s⁻) * Σ * LinearAlgebra.Diagonal(s⁻)
    for i in axes(C, 1)
        C[i, i] = s[i] > 0 ? one(eltype(C)) : zero(eltype(C))
    end
    return C
end

"""
    nobs(fr; dataset = nothing)

Number of observations; for a multi-dataset fit, `dataset = i` restricts to
one dataset. Same keyword on [`residuals`](@ref) and [`rss`](@ref).
"""
function StatsAPI.nobs(fr::FitResult; dataset = nothing)
    dataset === nothing && return length(fr.residuals)
    return length(_dataset_range(fr, dataset))
end

# A zero-width marginal is a parameter pinned by a `Dirac` prior (`Normal(μ, 0)`
# from least squares, `Dirac` from MCMC); it was not fitted and consumes no
# degree of freedom.
_n_free(fr::FitResult) = count(d -> Statistics.std(d) > 0, getdata(fr.marginals))

"""
    dof(fr)

Degrees of freedom consumed by the model (StatsAPI semantics): the fitted
parameters, plus the noise scale when it had to be estimated (no per-point
uncertainties). Parameters pinned by a zero-width prior are not counted. For
the residual degrees of freedom use [`dof_residual`](@ref).
"""
StatsAPI.dof(fr::FitResult) =
    _n_free(fr) + (StatsAPI.weights(fr) === nothing ? 1 : 0)

"""
    dof_residual(fr)

Residual degrees of freedom: `nobs(fr)` minus the fitted parameters. Pinned
parameters are not counted.
"""
StatsAPI.dof_residual(fr::FitResult) = StatsAPI.nobs(fr) - _n_free(fr)

"""
    rss(fr; dataset = nothing)

Residual sum of squares. With per-point uncertainties the residuals are
σ-normalised, so this is χ².
"""
StatsAPI.rss(fr::FitResult; dataset = nothing) =
    sum(abs2, StatsAPI.residuals(fr; dataset))
StatsAPI.weights(fr::FitResult) = fr.weights

"""
    residuals(fr; dataset = nothing)

Data residuals `data - model`, divided by σᵢ when the data carried per-point
uncertainties. Prior rows are not included.
"""
function StatsAPI.residuals(fr::FitResult; dataset = nothing)
    dataset === nothing && return fr.residuals
    return fr.residuals[_dataset_range(fr, dataset)]
end

"""
    confint(fr; level = 0.95) -> (lower, upper)

Per-parameter equal-tailed interval from the marginal distributions, as two
`ComponentVector`s: `confint(fr).lower.ampl`, `confint(fr).upper.ampl`.
"""
function StatsAPI.confint(fr::FitResult; level::Real = 0.95)
    α = (1 - level) / 2
    return (
        lower = _component_func(fr.marginals, d -> Distributions.quantile(d, α)),
        upper = _component_func(fr.marginals, d -> Distributions.quantile(d, 1 - α)),
    )
end

"""
    predict(fr, input = fr.input; uncertainty = false, dataset = nothing)

Model evaluated at the fitted parameters. With `uncertainty = true` returns
`Normal`s whose σ is the delta-method confidence band `√diag(J Σ Jᵀ)` from
[`vcov`](@ref): model uncertainty only, no data noise. For a multi-dataset
fit, `dataset = i` evaluates that dataset's model (at its own input unless
`input` is given).
"""
function StatsAPI.predict(
        fr::FitResult, input = fr.input; uncertainty::Bool = false, dataset = nothing
    )
    model, x = if dataset === nothing
        fr.model, input
    else
        _dataset_range(fr, dataset)
        fr.model.models[dataset], (input === fr.input ? fr.input[dataset] : input)
    end
    θ = StatsAPI.coef(fr)
    pred = model(x, θ)
    uncertainty || return pred
    Σ = StatsAPI.vcov(fr)
    ax = ComponentArrays.getaxes(θ)
    J = ForwardDiff.jacobian(
        p -> model(x, ComponentArrays.ComponentVector(p, ax...)),
        ComponentArrays.getdata(θ),
    )
    # diag(J Σ Jᵀ) without forming the dense m×m matrix.
    σ = sqrt.(max.(0, sum((J * Σ) .* J; dims = 2)))
    return Distributions.Normal.(pred, reshape(σ, size(pred)))
end

"""
    loglikelihood(fr)

Gaussian log-likelihood at the fitted parameters. With known per-point σ:
`-χ²/2 + Σ log wᵢ - n/2 log 2π`. Without, the MLE noise scale `σ̂² = rss/n`
is plugged in and counts as one extra consumed degree of freedom in
[`dof`](@ref), and so in [`aic`](@ref)/[`bic`](@ref).
"""
function StatsAPI.loglikelihood(fr::FitResult)
    n = StatsAPI.nobs(fr)
    r = StatsAPI.rss(fr)
    w = StatsAPI.weights(fr)
    w === nothing && return -n / 2 * (log(2π * r / n) + 1)
    return -r / 2 + sum(log, w) - n / 2 * log(2π)
end

"""
    aic(fr)

Akaike information criterion, `2 dof(fr) - 2 loglikelihood(fr)`.
"""
StatsAPI.aic(fr::FitResult) = 2 * StatsAPI.dof(fr) - 2 * StatsAPI.loglikelihood(fr)

"""
    bic(fr)

Bayesian information criterion, `dof(fr) log nobs(fr) - 2 loglikelihood(fr)`.
"""
StatsAPI.bic(fr::FitResult) =
    StatsAPI.dof(fr) * log(StatsAPI.nobs(fr)) - 2 * StatsAPI.loglikelihood(fr)

"""
    mse(fr)

`rss(fr) / dof_residual(fr)`. Its meaning depends on the data:

- With per-point uncertainties, the residuals are divided by σᵢ and `mse` is
  the reduced χ², ≈ 1 for a good model (see [`reduced_chisq`](@ref)).
- Without, `mse` is the estimated noise variance σ̂², in squared data units.
"""
mse(fr::FitResult) = StatsAPI.rss(fr) / StatsAPI.dof_residual(fr)

"""
    reduced_chisq(fr)

Reduced χ²: `rss(fr) / dof_residual(fr)` with σ-normalised residuals. Errors
unless the fit had per-point uncertainties: without them the same ratio is a
variance estimate, not a goodness-of-fit statistic (use [`mse`](@ref)).
"""
function reduced_chisq(fr::FitResult)
    StatsAPI.weights(fr) === nothing && error(
        "reduced_chisq requires per-point data uncertainties, but this fit had none " *
        "(weights(fr) === nothing). Pass `data` as distributions, e.g. `Normal.(y, σ)`. " *
        "For the estimated noise variance of an unweighted fit, use `mse`.")
    return StatsAPI.rss(fr) / StatsAPI.dof_residual(fr)
end

# "value ± σ" with σ rounded to two significant digits and the value to match.
function _pm_string(μ::Real, σ::Real)
    iszero(σ) && return string(round(μ, sigdigits = 6), " (fixed)")
    (isfinite(σ) && isfinite(μ)) || return string(μ, " ± ", σ)
    σ = round(σ; sigdigits = 2)
    dig = 1 - floor(Int, log10(σ))   # negative for σ ≥ 100
    return string(round(μ; digits = dig), " ± ", round(σ; digits = dig))
end
_pm_string(d) = _pm_string(Statistics.mean(d), Statistics.std(d))

function _push_lines!(lines, name, v)
    if v isa ComponentVector
        for k in keys(v)
            _push_lines!(lines, string(name, ".", k), v[k])
        end
    elseif v isa AbstractArray
        for (i, d) in pairs(v)
            idx = i isa CartesianIndex ? join(Tuple(i), ",") : string(i)
            push!(lines, (string(name, "[", idx, "]"), _pm_string(d)))
        end
    else
        push!(lines, (string(name), _pm_string(v)))
    end
    return lines
end

function _param_lines(fr::FitResult)
    lines = Tuple{String, String}[]
    for k in keys(fr.marginals)
        _push_lines!(lines, string(k), fr.marginals[k])
    end
    return lines
end

Base.show(io::IO, fr::FitResult) =
    print(io, "FitResult(", join((string(l, " = ", s) for (l, s) in _param_lines(fr)), ", "), ")")

function Base.show(io::IO, ::MIME"text/plain", fr::FitResult)
    conv = fr.diagnostics === nothing ? "" :
        fr.diagnostics.converged ? ", converged" : ", NOT converged"
    nds = fr.model isa DatasetStack ?
        string(length(fr.model.models), " datasets, ") : ""
    println(io, "FitResult (", nds, StatsAPI.nobs(fr), " observations", conv, ")")
    lines = _param_lines(fr)
    pad = maximum(length(first(l)) for l in lines)
    for (i, (l, s)) in enumerate(lines)
        print(io, "  ", rpad(l, pad), " = ", s)
        i < length(lines) && println(io)
    end
end
