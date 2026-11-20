"""
    DriftModel

How tracked parameters evolve along the series coordinate. A drift model
implements three functions:

- `init_drift(drift, keys, priors) -> state`
- `drift_predict(state, key, t) -> (μ, σ)`: same shape as the component
- `drift_update!(state, key, t, μ, σ)`
"""
abstract type DriftModel end

"""
    init_drift(drift, keys, priors) -> state

Drift state for the tracked components `keys`, initialised from `priors`.
"""
function init_drift(drift::DriftModel, keys, priors)
    drift isa GaussianDrift && error("GaussianDrift needs `using AbstractGPs`.")
    error("init_drift not implemented for $(typeof(drift)). See `DriftModel`.")
end

"""
    drift_predict(state, key, t) -> (μ, σ)

Predictive mean and standard deviation of component `key` at coordinate `t`,
each the same shape as the component.
"""
function drift_predict end

"""
    drift_update!(state, key, t, μ, σ)

Condition component `key` on a fitted value `μ ± σ` at coordinate `t`.
"""
function drift_update! end

"""
    GaussianDrift(; lengthscale, kernel = nothing, noise_inflation = 1.0, window = 64)

GP drift: each tracked scalar gets `GP(ConstMean(μ₀), σ₀²·k)`, mean and variance
taken from its prior. Default kernel `½·SE(ℓ) + ½·SE(10ℓ)`; `kernel` accepts any
unit-variance KernelFunctions kernel instead. Only components with `Normal`
priors can be tracked.

Each converged frame conditions the GP on what its data alone say: the frame's
prior, which is the GP's own prediction, is divided out of its posterior first
(per scalar, as Gaussians). The observation noise is `(noise_inflation·σ)²`;
`noise_inflation > 1` is a safety margin against model error. Only the last `window`
observations per scalar are kept, so the cost stays O(window³).

Requires `using AbstractGPs`.
"""
Base.@kwdef struct GaussianDrift{L, K} <: DriftModel
    lengthscale::L
    kernel::K = nothing
    noise_inflation::Float64 = 1.0
    window::Int = 64
end

"""
    StaticDrift()

No drift: every frame keeps the initial prior and results never feed back, i.e.
sequential fitting with warm starts only. The default, and needs no extension.
Use [`GaussianDrift`](@ref) for GP filtering.
"""
struct StaticDrift <: DriftModel end

struct StaticDriftState{P}
    priors::P
end

init_drift(::StaticDrift, keys, priors) = StaticDriftState(priors)
drift_predict(st::StaticDriftState, key::Symbol, t) = (
    _component_apply(Statistics.mean, st.priors[key]),
    _component_apply(Statistics.std, st.priors[key]),
)
drift_update!(::StaticDriftState, key, t, μ, σ) = nothing

drift_prior(state, key, t, original) =
    Distributions.Normal.(drift_predict(state, key, t)...)
drift_prior(st::StaticDriftState, key, t, original) = original

"""
    Tracked(keys...) / NotTracked(keys...)

Selectors for which components of the priors drift along the series. With no
selector, every trackable component drifts. `Tracked` lists exactly the drifting
components; `NotTracked` lists the static ones and everything else drifts.
Components pinned by zero-width priors, or whose priors are not distributions,
are never tracked.

A drift model other than [`StaticDrift`](@ref) replaces the prior of a tracked
component by a `Normal`, so it tracks only components whose priors are
`Normal`: others (a `Uniform` box) keep their prior in every frame, and naming
one in `Tracked` is an error.
"""
struct Tracked
    keys::Vector{Symbol}
end
Tracked(ks::Symbol...) = Tracked(collect(ks))

struct NotTracked
    keys::Vector{Symbol}
end
NotTracked(ks::Symbol...) = NotTracked(collect(ks))
@doc (@doc Tracked) NotTracked

_trackable(v::Distributions.Distribution) = Statistics.std(v) > 0
_trackable(v::AbstractArray) = all(_trackable, v)
_trackable(v) = false

_all_normal(v::Distributions.Normal) = true
_all_normal(v::AbstractArray) = all(_all_normal, v)
_all_normal(v) = false

# A drift model hands back `Normal` priors (`drift_prior`), which would turn a
# tracked `Uniform` box into a soft prior without its bounds.
_drift_trackable(::DriftModel, v) = _all_normal(v)
_drift_trackable(::StaticDrift, v) = true

function _tracked_keys(track, priors, drift::DriftModel)
    tkeys = _tracked_keys(track, priors)
    if track isa Tracked
        for k in tkeys
            _drift_trackable(drift, priors[k]) || error(
                "component $k cannot be tracked by $(nameof(typeof(drift))): its prior is " *
                "not `Normal`. Leave it out of `Tracked`; it keeps its prior in every frame.")
        end
        return tkeys
    end
    return [k for k in tkeys if _drift_trackable(drift, priors[k])]
end

function _tracked_keys(track, priors)
    ks = collect(keys(priors))
    if track isa Tracked || track isa NotTracked
        bad = setdiff(track.keys, ks)
        isempty(bad) || error("track selector names unknown component(s) $bad; " *
                              "available: $ks.")
    end
    if track === nothing
        return [k for k in ks if _trackable(priors[k])]
    elseif track isa Tracked
        for k in track.keys
            _trackable(priors[k]) || error(
                "component $k cannot be tracked: its prior is pinned or not a distribution.")
        end
        return track.keys
    elseif track isa NotTracked
        return [k for k in setdiff(ks, track.keys) if _trackable(priors[k])]
    end
    error("`track` must be `Tracked(...)`, `NotTracked(...)`, or `nothing`.")
end

"""
    SeriesFitResult

Result of [`fit_series`](@ref): the coordinate vector `ts`, one
[`FitResult`](@ref) per frame in `frames`, and the final drift state. A frame
that failed outright holds its prior as the marginals, with
`converged == false`. Index with `sres[k]`; summarise with
[`trajectory`](@ref), `coef`, `stderror`, `converged`.
"""
struct SeriesFitResult{T, S, D}
    ts::T
    frames::S
    drift_state::D
end

Base.getindex(s::SeriesFitResult, k::Integer) = s.frames[k]
Base.length(s::SeriesFitResult) = length(s.frames)
Base.iterate(s::SeriesFitResult, i = 1) =
    i > length(s.frames) ? nothing : (s.frames[i], i + 1)

converged(s::SeriesFitResult) = [converged(fr) for fr in s.frames]

StatsAPI.coef(s::SeriesFitResult) =
    reduce(vcat, (permutedims(collect(ComponentArrays.getdata(StatsAPI.coef(fr)))) for fr in s.frames))
StatsAPI.stderror(s::SeriesFitResult) =
    reduce(vcat, (permutedims(collect(ComponentArrays.getdata(StatsAPI.stderror(fr)))) for fr in s.frames))

"""
    trajectory(sres, key; level = 0.95) -> (t, mean, std, lower, upper)

One component's evolution along the series, with equal-tailed `level`
intervals from each frame's marginals (posterior quantiles for MCMC
frames, Gaussian ones for LM frames).
"""
function trajectory(s::SeriesFitResult, key::Symbol; level::Real = 0.95)
    α = (1 - level) / 2
    grab(f) = [_component_apply(f, fr.marginals[key]) for fr in s.frames]
    return (
        t = s.ts,
        mean = grab(Statistics.mean),
        std = grab(Statistics.std),
        lower = grab(d -> Distributions.quantile(d, α)),
        upper = grab(d -> Distributions.quantile(d, 1 - α)),
    )
end

function Base.show(io::IO, s::SeriesFitResult)
    nc = count(converged(s))
    print(io, "SeriesFitResult(", length(s), " frames, ", nc, " converged)")
end

# Shared input vs one input per frame. By default a vector of arrays is
# per-frame, anything else (one grid, a range, a tuple of grids) is shared;
# `per_frame` overrides the guess.
_per_frame(inputs, n) = begin
    length(inputs) == n || error(
        "got $(length(inputs)) per-frame inputs for $n frames.")
    inputs
end
_frame_inputs(inputs::AbstractVector{<:AbstractArray}, n, ::Nothing) = _per_frame(inputs, n)
_frame_inputs(inputs, n, ::Nothing) = fill(inputs, n)
_frame_inputs(inputs, n, per_frame::Bool) = per_frame ? _per_frame(inputs, n) : fill(inputs, n)

# Errors that point at the code, not the frame. `fit_series` rethrows them.
# MCMC chains run in tasks, which wrap errors.
const _CODE_ERRORS = @static if isdefined(Core, :FieldError)
    Union{InterruptException, MethodError, UndefVarError, UndefKeywordError, Core.FieldError}
else
    Union{InterruptException, MethodError, UndefVarError, UndefKeywordError}
end
_code_error(err) = err isa _CODE_ERRORS
_code_error(err::CompositeException) = any(_code_error, err.exceptions)
_code_error(err::TaskFailedException) = _code_error(err.task.exception)

# The frame's posterior with its prior divided out (per scalar, Gaussian): what
# the frame's data alone say.
function _likelihood_only(μ::Real, σ::Real, prior::Distributions.Normal)
    σ > 0 || return μ, σ
    p0 = inv(prior.σ^2)
    p = inv(σ^2) - p0
    p > 1.0e-12 * p0 || return float(μ), 1.0e6 * prior.σ   # data added nothing
    return (μ / σ^2 - prior.μ * p0) / p, inv(sqrt(p))
end
_likelihood_only(μ::Real, σ::Real, prior) = (μ, σ)
function _likelihood_only(μ::AbstractArray, σ::AbstractArray, prior::AbstractArray)
    res = map(_likelihood_only, μ, σ, prior)
    return map(first, res), map(last, res)
end

function _frame_priors(priors, tkeys, state, t)
    prs = [k => (k in tkeys ? drift_prior(state, k, t, priors[k]) : priors[k])
           for k in keys(priors)]
    return ComponentArrays.ComponentVector(; prs...)
end

"""
    fit_series(model, ts, inputs, data_per_frame, priors, method;
               drift, track = nothing, per_frame_inputs = nothing, kwargs...)

Fit an ordered series of datasets whose parameters drift smoothly along the
coordinate `ts` (time, position). A forward Bayesian filter: frames are fit in
order, each frame's prior for the tracked components is the drift model's
predictive at its coordinate, its start point is the previous frame's estimate,
and each converged result conditions the drift for the frames after it, on the
information of its own data only (the frame's prior is divided out).

- `inputs`: one shared input, or a vector with one input per frame. A vector of
  arrays is read as per-frame; `per_frame_inputs = true`/`false` says which it
  is when that guess is wrong (a shared input that is itself a vector of
  arrays).
- `data_per_frame`: one data vector per frame.
- `priors`: initial prior; for tracked components it also sets the drift's mean
  and variance scale. `track`: see [`Tracked`](@ref).
- `drift`: a [`DriftModel`](@ref). Default [`StaticDrift`](@ref), i.e. fixed
  priors and warm-started sequential fits; [`GaussianDrift`](@ref) for GP
  filtering.
- Remaining keywords go to every per-frame `fit_model` call.

A frame that fails or does not converge is warned about, recorded unconverged
(holding its prior as the marginals if the solve threw), and skipped by the
drift, which extrapolates over it. Errors that point at the code rather than
the frame (`MethodError`, `UndefVarError`, a missing field, Ctrl-C) are
rethrown.
"""
function fit_series(
        model, ts, inputs, data_per_frame, priors, method::FitMethod;
        drift::DriftModel = StaticDrift(), track = nothing, per_frame_inputs = nothing,
        kwargs...
    )
    n = length(ts)
    length(data_per_frame) == n || error("got $(length(data_per_frame)) data vectors for $n frames.")
    xs = _frame_inputs(inputs, n, per_frame_inputs)

    tkeys = _tracked_keys(track, priors, drift)
    state = init_drift(drift, tkeys, priors)

    frames = FitResult[]
    lastcoef = nothing
    for (k, t) in enumerate(ts)
        priors_k = _frame_priors(priors, tkeys, state, t)
        # Frame 1 starts at its prior mean: deterministic, and keeps samplers
        # out of mirror modes; later frames at the previous estimate.
        init = lastcoef === nothing ? map(Statistics.mean, priors_k) : lastcoef
        threw = false
        fr = try
            fit_model(model, xs[k], data_per_frame[k], priors_k, method; init, kwargs...)
        catch err
            _code_error(err) && rethrow()
            threw = true
            @warn "frame $k (t = $t) failed; recording its prior, drift skips it." exception = err
            FitResult(model, xs[k], priors_k, Float64[], nothing, nothing,
                (; converged = false, error = err))
        end
        push!(frames, fr)
        if converged(fr)
            lastcoef = StatsAPI.coef(fr)
            se = StatsAPI.stderror(fr)
            for key in tkeys
                μl, σl = _likelihood_only(StatsAPI.coef(fr)[key], se[key], priors_k[key])
                drift_update!(state, key, t, μl, σl)
            end
        elseif !threw
            @warn "frame $k (t = $t) did not converge; drift not conditioned on it."
        end
    end
    return SeriesFitResult(ts, frames, state)
end
