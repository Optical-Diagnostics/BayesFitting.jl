module BayesFittingNonlinearSolveExt
using BayesFitting, NonlinearSolve

using Distributions
using ComponentArrays

using FiniteDiff, ForwardDiff, LinearAlgebra
using ForwardDiff: DiffResults

import BayesFitting: LevenbergMarquardtFit, fit_model, FitResult, inverse_sigmas
import BayesFitting: ParameterScaling, scaled, unscaled, unscaled_cov, flatten_priors, init_point


prior_residual(d, x) = error("Regularization term not defined for prior distribution: $(typeof(d))")

prior_residual(d::Dirac, x) = zero(x)
# A zero-width `Normal` pins its parameter like a `Dirac`: `x` sits exactly at
# `μ`, and `(x - μ) / σ` would be 0/0.
prior_residual(d::Normal, x) = iszero(d.σ) ? zero(x) : (x - d.μ) / d.σ

# Wall stiffness for `Uniform` priors, in residual units per box width of
# violation.
const WALL_STIFFNESS = 1.0e3

# Zero inside the box, proportional to the violation outside, in units of the
# box width. The model is evaluated at the clamped parameters, so outside the
# box this row alone depends on the parameter and leads the solver back.
prior_residual(d::Uniform, x) =
    WALL_STIFFNESS * (x - clamp(x, d.a, d.b)) / (d.b - d.a)


# `walls = false` leaves out the `Uniform` boxes, for the covariance.
function weighted_residual(model, input, data, params, w, ps::ParameterScaling; walls::Bool = true)
    y = mean.(data)
    ax = getaxes(params)
    lo, hi = box_limits(params, walls)
    dists, index = walls ? flatten_priors(params) : soft_priors(params)
    n = length(y)
    m = length(dists)
    return function (z, _)
        θ_raw = unscaled(ps, z)
        θ = ComponentVector(clamp.(θ_raw, lo, hi), ax...)
        pred = model(input, θ)
        check_prediction(pred, n)
        out = Vector{eltype(pred)}(undef, n + m)
        @inbounds @simd for i in 1:n
            out[i] = (y[i] - pred[i]) * w[i]
        end
        @inbounds for t in 1:m
            out[n + t] = prior_residual(dists[t], θ_raw[index[t]])
        end
        return out
    end
end

# Residual for data without uncertainty: `yᵢ - model`, then one row per
# `Uniform` bound. `Normal` priors are dropped, since there is no noise scale
# to weigh them against. The wall is scaled by the RMS of the data.
function unweighted_residual(model, input, data, params, ps::ParameterScaling; walls::Bool = true)
    y = collect(data)
    ax = getaxes(params)
    lo, hi = box_limits(params, walls)
    dists, index = walls ? box_priors(params) : (Uniform{Float64}[], Int[])
    n = length(y)
    m = length(dists)
    rms = sqrt(sum(abs2, y) / max(n, 1))
    wall = iszero(rms) ? one(rms) : rms
    return function (z, _)
        θ_raw = unscaled(ps, z)
        θ = ComponentVector(clamp.(θ_raw, lo, hi), ax...)
        pred = model(input, θ)
        check_prediction(pred, n)
        out = Vector{eltype(pred)}(undef, n + m)
        @inbounds @simd for i in 1:n
            out[i] = y[i] - pred[i]
        end
        @inbounds for t in 1:m
            out[n + t] = wall * prior_residual(dists[t], θ_raw[index[t]])
        end
        return out
    end
end

# The residual loops read the prediction under `@inbounds`.
function check_prediction(pred, n)
    Base.require_one_based_indexing(pred)
    length(pred) == n || throw(DimensionMismatch(
        "model output has length $(length(pred)), data has length $n"))
    return nothing
end

function box_limits(params, walls::Bool = true)
    flat = getdata(params)
    lo = [walls && d isa Uniform ? float(d.a) : -Inf for d in flat]
    hi = [walls && d isa Uniform ? float(d.b) : Inf for d in flat]
    return lo, hi
end

function soft_priors(params)
    dists, index = flatten_priors(params)
    keep = findall(d -> !(d isa Uniform), dists)
    return identity.(dists[keep]), index[keep]
end

function box_priors(params)
    dists, index = flatten_priors(params)
    keep = findall(d -> d isa Uniform, dists)
    return identity.(dists[keep]), index[keep]
end

# Labels of the parameters on a `Uniform` bound at `θ` (flat, physical units,
# clamped into the box), to within 1e-9 of the box width.
function parameters_at_bound(params, θ)
    dists, index = box_priors(params)
    hit = Int[]
    for (d, i) in zip(dists, index)
        tol = 1.0e-9 * (d.b - d.a)
        d.a + tol < θ[i] < d.b - tol || push!(hit, i)
    end
    return ComponentArrays.labels(params)[hit]
end

# NonlinearSolve's LM computes the Jacobian at each accepted point in a
# separate ForwardDiff pass. `FusedResidual` runs one dual pass per residual
# call, returns the value and caches the Jacobian for the `jac` call that
# follows at the same point.

mutable struct FusedResidual{F, G, U, R, C}
    const residual::F
    const residual_1arg::G   # single-argument form, the one `jacobian_config` is built for
    const cached_point::U
    cache_valid::Bool
    const jacobian_buffer::R
    const jacobian_config::C
end

function FusedResidual(f, z0)
    g = Base.Fix2(f, nothing)
    y0 = g(z0)   # probe evaluation, only to size the buffers
    return FusedResidual(
        f, g, fill!(similar(z0), NaN), false,
        DiffResults.JacobianResult(y0, z0), ForwardDiff.JacobianConfig(g, z0)
    )
end

function (fr::FusedResidual)(z::AbstractVector{<:AbstractFloat}, _)
    ForwardDiff.jacobian!(fr.jacobian_buffer, fr.residual_1arg, z, fr.jacobian_config)
    copyto!(fr.cached_point, z)
    fr.cache_valid = true
    # Copied: the solver keeps the reference while the buffer is overwritten
    # by the next trial evaluation.
    return copy(DiffResults.value(fr.jacobian_buffer))
end

# Dual-valued parameters (post-fit Hessian) bypass the fused machinery.
(fr::FusedResidual)(z, q) = fr.residual(z, q)

raw_residual(f) = f
raw_residual(fr::FusedResidual) = fr.residual

residual_jacobian(f, z) = ForwardDiff.jacobian(Base.Fix2(f, nothing), z)
function residual_jacobian(fr::FusedResidual, z)
    fr.cache_valid && z == fr.cached_point && return DiffResults.jacobian(fr.jacobian_buffer)
    return ForwardDiff.jacobian(fr.residual_1arg, z)
end

function fused_jacobian(fr::FusedResidual)
    return function (z, _)
        # Copied for the same reason as the value: damping retries reuse the
        # returned J after later trial evaluations refill the buffer.
        fr.cache_valid && z == fr.cached_point && return copy(DiffResults.jacobian(fr.jacobian_buffer))
        return ForwardDiff.jacobian(fr.residual_1arg, z)
    end
end

function _problem(resid, fuse::Bool, z0)
    fuse || return NonlinearLeastSquaresProblem(NonlinearFunction(resid), z0)
    fr = FusedResidual(resid, z0)
    return NonlinearLeastSquaresProblem(NonlinearFunction(fr; jac = fused_jacobian(fr)), z0)
end

# `init` is the start point in physical units, by default the prior means.
function create_nonlinear_problem(
        model, input, data, params, w = inverse_sigmas(data);
        fuse::Bool = false, init = nothing, ps::ParameterScaling = ParameterScaling(params)
    )
    z0 = scaled(ps, init === nothing ? params : init_point(init, params))
    if w === nothing
        if any(p -> p isa Distribution && !(p isa Uniform) && !iszero(std(p)), params)
            @warn "No data uncertainty supplied; priors on the parameters are ignored " *
                  "(`Uniform` bounds and pinned parameters are kept). " *
                  "Pass `data` as distributions (e.g. `Normal.(y, σ)`) to use them."
        end
        return _problem(unweighted_residual(model, input, data, params, ps), fuse, z0)
    else
        return _problem(weighted_residual(model, input, data, params, w, ps), fuse, z0)
    end
end


# Invert the Hessian of the negative log posterior over the `free` parameters.
# Returns `(σ, Σ, ok)` in scaled parameters. Pinned parameters get zero
# variance. Warns, with `ok = false`, on a negative eigenvalue (a saddle), on
# ill-conditioning and on a non-finite `H`. A direction the data cannot
# constrain gets a variance of `1/tol`.
function parameter_covariance(H, free)
    n = size(H, 1)
    T = float(eltype(H))
    Σ = zeros(T, n, n)
    any(free) || return zeros(T, n), Σ, true
    if !all(isfinite, view(H, free, free))
        @warn "Hessian is not finite at the solution (a model derivative is infinite or " *
              "NaN there); no uncertainties are available."
        Σ[free, free] .= T(NaN)
        return ifelse.(free, T(Inf), zero(T)), Σ, false
    end
    (; values, vectors) = eigen(Symmetric(Matrix{T}(H[free, free])))
    amax = maximum(abs, values)
    amin = minimum(abs, values)
    # Round-off puts the null eigenvalues of JᵀJ at about ±eps·amax: neither a
    # saddle nor a usable curvature. `floatmin` keeps an all-zero block finite.
    tol = max(eps(T) * count(free) * amax, floatmin(T))
    w = map(λ -> abs(λ) > tol ? inv(λ) : inv(tol), values)
    Σfree = vectors * Diagonal(w) * vectors'
    Σ[free, free] .= (Σfree .+ Σfree') ./ 2
    ok = true
    nneg = count(<(-tol), values)
    if nneg > 0
        ok = false
        @warn "Hessian is not positive definite at the solution: the fit is at a saddle " *
              "or has not converged. Reported uncertainties are unreliable." negative_eigenvalues = nneg
    end
    κ = amin > 0 && isfinite(amax) ? amax / amin : T(Inf)
    if κ > 1 / sqrt(eps(T))
        ok = false
        @warn "Hessian is ill-conditioned: some parameter combination is barely constrained " *
              "by the data." cond = κ
    end
    return sqrt.(abs.(diag(Σ))), Σ, ok
end

function fit_model(
    model, input, data, params, method::LevenbergMarquardtFit;
    disable_geodesic=Val(true), autodiff=AutoForwardDiff(), show_trace=Val(false),
    trace_level=TraceWithJacobianConditionNumber(), maxiters=1000,
    abstol=1.0e-6, reltol=1.0e-3,
    termination_condition=AbsNormSafeBestTerminationMode(Base.Fix2(norm, 2); patience_steps=3, max_stalled_steps=3),
    damping_initial=1.0, damping_increase_factor=2.0, damping_decrease_factor=3.0,
    alg=LevenbergMarquardt(;
        disable_geodesic, autodiff,
        damping_initial, damping_increase_factor, damping_decrease_factor),
    # Fusing needs ForwardDiff and no geodesic acceleration.
    fuse=autodiff isa AutoForwardDiff && disable_geodesic isa Val{true},
    approximate_uncertainties=false,
    init=nothing,
    scale=nothing
)

    w = inverse_sigmas(data)
    ps = ParameterScaling(params; scale)
    # Without per-point σ the noise scale is estimated from the residuals,
    # which needs more observations than fitted parameters.
    if w === nothing && length(data) <= count(ps.free)
        throw(ArgumentError(
            "$(length(data)) observations cannot determine $(count(ps.free)) free parameters " *
            "and the noise level. Supply per-point uncertainties (`Normal.(y, σ)`), " *
            "pin parameters with `Dirac`, or use more data."))
    end
    prob = create_nonlinear_problem(model, input, data, params, w; fuse, init, ps)

    sol = solve(prob, alg; show_trace, abstol, reltol, termination_condition, trace_level, maxiters)

    solver_ok = NonlinearSolve.SciMLBase.successful_retcode(sol.retcode)
    solver_ok || @warn "Solver did not converge; parameters are the best point found." retcode = sol.retcode

    nprior = length(first(w === nothing ? box_priors(params) : flatten_priors(params)))
    ndata = length(sol.resid) - nprior
    data_resid = sol.resid[1:ndata]

    # Estimate in physical units, clamped into the `Uniform` boxes: a wall
    # brings the solver back to the bound from just outside it.
    lo, hi = box_limits(params)
    θ = clamp.(unscaled(ps, sol.u), lo, hi)
    bounded = any(isfinite, lo) || any(isfinite, hi)
    # The covariance is that of the objective without the walls, at the
    # estimate.
    z_at_estimate = bounded ? BayesFitting._scaled.(θ, ps.μ, ps.σ) : sol.u
    rawf = if bounded
        w === nothing ? unweighted_residual(model, input, data, params, ps; walls = false) :
            weighted_residual(model, input, data, params, w, ps; walls = false)
    else
        raw_residual(sol.prob.f.f)
    end
    # Known per-point σ: residuals are already normalised (σ² = 1). Otherwise the
    # noise scale is estimated from the fit, σ̂² = RSS/(n - p), p the free parameters.
    σ² = w === nothing ? sum(abs2, data_resid) / (ndata - count(ps.free)) : one(eltype(data_resid))

    H = if approximate_uncertainties
        # Drops the residual-curvature term Σᵢ rᵢ ∇²rᵢ: exact for models linear
        # in the parameters, and at zero residual.
        J = residual_jacobian(bounded ? rawf : sol.prob.f.f, z_at_estimate)
        # Finite differences step off the estimate, so not for one on a bound.
        if any(!isfinite, J) && !bounded
            J = FiniteDiff.finite_difference_jacobian(Base.Fix2(rawf, nothing), z_at_estimate)
        end
        (J' * J) / σ²
    else
        nlp = θ -> sum(abs2, rawf(θ, nothing)) / 2 / σ²
        Hexact = ForwardDiff.hessian(nlp, z_at_estimate)
        if any(!isfinite, Hexact) && !bounded
            Hexact = FiniteDiff.finite_difference_hessian(nlp, z_at_estimate)
        end
        Hexact
    end

    std_fit, Σscaled, cov_ok = parameter_covariance(H, ps.free)

    Σ = unscaled_cov(ps, Σscaled)

    at_bound = parameters_at_bound(params, θ)
    isempty(at_bound) || @warn "Parameters at a `Uniform` bound. Their uncertainties " *
        "are those of the unbounded problem at that point and ignore the bound." at_bound

    return FitResult(
        model, input,
        ComponentVector(Normal.(θ, ps.σ .* std_fit), ps.axes...),
        data_resid,
        w,
        Σ,
        (; converged = solver_ok && cov_ok, retcode = sol.retcode, covariance_ok = cov_ok, at_bound),
    )
end

end
