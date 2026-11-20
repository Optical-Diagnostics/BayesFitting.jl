module FittingUtilsAbstractGPsExt

using FittingUtils, AbstractGPs
using Statistics
import FittingUtils: GaussianDrift, init_drift, drift_predict, drift_update!

# The posterior is rebuilt from the window on each predict.
mutable struct ScalarState{F}
    const gp::F
    const t::Vector{Float64}
    const y::Vector{Float64}
    const noise_var::Vector{Float64}
    # Floor for those variances, a tiny fraction of the prior variance: a zero
    # σ (degenerate frame) would otherwise make the GP interpolate exactly.
    const noise_floor::Float64
end

function ScalarState(cfg::GaussianDrift, μ0::Real, σ0::Real)
    base = cfg.kernel === nothing ?
        0.5 * with_lengthscale(SqExponentialKernel(), cfg.lengthscale) +
        0.5 * with_lengthscale(SqExponentialKernel(), 10 * cfg.lengthscale) :
        cfg.kernel
    f = GP(ConstMean(float(μ0)), float(σ0)^2 * base)
    return ScalarState(f, Float64[], Float64[], Float64[], 1.0e-12 * float(σ0)^2)
end

function _predict(s::ScalarState, t)
    fp = isempty(s.t) ? s.gp : posterior(s.gp(s.t, s.noise_var), s.y)
    # The prediction noise is the relative floor. AbstractGPs' absolute default
    # of 1e-18 is too large for a parameter of size 1e-9.
    fx = fp([float(t)], s.noise_floor)
    return only(mean(fx)), sqrt(only(var(fx)))
end

function _update!(s::ScalarState, cfg::GaussianDrift, t, μ, σ)
    push!(s.t, float(t))
    push!(s.y, float(μ))
    push!(s.noise_var, max((cfg.noise_inflation * float(σ))^2, s.noise_floor))
    if length(s.t) > cfg.window
        popfirst!(s.t); popfirst!(s.y); popfirst!(s.noise_var)
    end
    return s
end

struct GPDriftState{C <: GaussianDrift}
    config::C
    components::Dict{Symbol, Any}   # key => ScalarState or same-shaped array of them
end

_states(cfg, d) = ScalarState(cfg, mean(d), std(d))
_states(cfg, a::AbstractArray) = map(x -> _states(cfg, x), a)

function FittingUtils.init_drift(drift::GaussianDrift, keys, priors)
    comps = Dict{Symbol, Any}(k => _states(drift, priors[k]) for k in keys)
    return GPDriftState(drift, comps)
end

function FittingUtils.drift_predict(st::GPDriftState, key::Symbol, t)
    c = st.components[key]
    c isa ScalarState && return _predict(c, t)
    res = map(s -> _predict(s, t), c)
    return first.(res), last.(res)
end

function FittingUtils.drift_update!(st::GPDriftState, key::Symbol, t, μ, σ)
    c = st.components[key]
    if c isa ScalarState
        _update!(c, st.config, t, μ, σ)
    else
        foreach((s, m, sd) -> _update!(s, st.config, t, m, sd), c, μ, σ)
    end
    return nothing
end

end
