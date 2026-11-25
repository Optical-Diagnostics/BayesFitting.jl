module BayesFittingTuringExt

using BayesFitting, Turing
import BayesFitting: TuringFit, fit_model, FitResult, _component_func, inverse_sigmas
import BayesFitting: ParameterScaling, scaled, unscaled, flatten_priors, init_point

using ComponentArrays
using Distributions
using Statistics
using LinearAlgebra
using StatsBase, EmpiricalDistributions


scaled_prior(d::Normal, μ, σ) = Normal((d.μ - μ) / σ, d.σ / σ)
scaled_prior(d::Uniform, μ, σ) = Uniform((d.a - μ) / σ, (d.b - μ) / σ)
scaled_prior(d, μ, σ) = error("MCMC supports Normal, Uniform, and Dirac priors, got $(typeof(d))")

# Free-parameter draws `z` into the full scaled vector; pinned entries are
# zero, which `unscaled` maps to the prior mean.
function full_scaled(z, free_idx, n)
    zfull = zeros(eltype(z), n)
    zfull[free_idx] .= z
    return zfull
end

@model function create_turing_model(model, input, data::AbstractArray{S}, ps, free_idx, zpriors) where {S<:Distribution}
    z ~ product_distribution(zpriors)

    θ = ComponentVector(unscaled(ps, full_scaled(z, free_idx, length(ps.μ))), ps.axes...)
    μ = model(input, θ)
    Turing.@addlogprob! logpdf(product_distribution(data), μ)
end

# Bin count for the histogram of `n` posterior samples. A bin width `h` inflates
# the variance by `h²/12`, so `√n` bins are used to keep the bias of `stderror`
# small.
posterior_bins(n) = clamp(round(Int, sqrt(n)), 50, 1000)

# Split-R̂ above this marks the chains as unconverged. 1.05 is between the
# classical Gelman-Rubin 1.1 and the strict 1.01 of Vehtari et al. (2021).
const RHAT_THRESHOLD = 1.05

# Split-R̂ (Gelman et al., BDA3) of one parameter's draws, `x` indexed as
# (iterations, chains). Each chain is split in half. `NaN` for fewer than 4
# iterations.
function split_rhat(x::AbstractMatrix)
    n = size(x, 1) ÷ 2
    n < 2 && return NaN
    halves = vec([view(x, (1:n) .+ (h - 1) * n, c) for h in 1:2, c in axes(x, 2)])
    W = mean(var.(halves))
    B = n * var(mean.(halves))
    return sqrt((W * (n - 1) / n + B / n) / W)
end

posterior_marginal(x) = UvBinnedDist(fit(Histogram, x; nbins = posterior_bins(length(x))))

function fit_model(
    model, input, data, params, method::TuringFit;
    sampler=NUTS(), num_samples=1000, num_chains=1, progress=true, init=nothing, scale=nothing
)
    # The likelihood is `data` itself, so it must carry the per-point σ.
    w = inverse_sigmas(data)
    w === nothing && throw(ArgumentError(
        "TuringFit needs per-point data uncertainties: pass `data` as " *
        "distributions, e.g. `Normal.(y, σ)`. LevenbergMarquardtFit also accepts " *
        "plain numbers and estimates the noise."))
    ps = ParameterScaling(params; scale)
    dists, index = flatten_priors(params)
    length(index) == length(params) || error("every parameter needs a prior distribution for MCMC")
    free_idx = [i for i in index if ps.free[i]]
    isempty(free_idx) && throw(ArgumentError(
        "every parameter is pinned by a zero-width prior: nothing to sample."))
    zpriors = [scaled_prior(d, ps.μ[i], ps.σ[i]) for (d, i) in zip(dists, index) if ps.free[i]]
    n = length(ps.μ)

    # All chains start at `init`, by default the prior means. Identical starts
    # weaken the between-chain check of split-R̂.
    start = init === nothing ? params : init_point(init, params)
    z0 = scaled(ps, start)[free_idx]
    # A least-squares start may sit exactly on a `Uniform` bound, where the
    # sampler's unconstraining transform is infinite: move it just inside.
    z0 = map(z0, zpriors) do z, d
        d isa Uniform ? clamp(z, d.a + 1e-6 * (d.b - d.a), d.b - 1e-6 * (d.b - d.a)) : z
    end
    init_kw = (; initial_params = InitFromParams((; z = z0)))

    # One model copy per chain: a model may hold caches, which parallel chains
    # would otherwise share.
    models = [c == 1 ? model : deepcopy(model) for c in 1:num_chains]
    chains = Vector{Any}(undef, num_chains)
    Threads.@threads for c in 1:num_chains
        posterior = create_turing_model(models[c], input, data, ps, free_idx, zpriors)
        chains[c] = sample(posterior, sampler, num_samples; progress = progress && c == 1, init_kw...)
    end

    # Draws in physical units, one (iterations, chains) matrix per flat
    # parameter; pinned parameters are constant at their prior mean.
    draws = map(1:n) do i
        j = findfirst(==(i), free_idx)
        j === nothing && return fill(ps.μ[i], num_samples, num_chains)
        reduce(hcat, [vec(getindex.(chains[c][:z], j)) for c in 1:num_chains]) .* ps.σ[i] .+ ps.μ[i]
    end
    samples = reduce(hcat, vec.(draws))
    marginals = [ps.free[i] ? posterior_marginal(view(samples, :, i)) : Dirac(ps.μ[i]) for i in 1:n]
    ps_out = ComponentVector(marginals, ps.axes)
    Σ = Matrix(Symmetric(cov(samples)))

    pred = model(input, _component_func(ps_out, mean))
    resid = (mean.(data) .- pred) .* w

    rh = [ps.free[i] ? split_rhat(draws[i]) : 1.0 for i in 1:n]
    mcmc_ok = maximum(rh) < RHAT_THRESHOLD
    mcmc_ok || @warn "MCMC chains have not mixed; posterior summaries are unreliable. " *
                     "Increase num_samples or reparametrise." max_rhat = maximum(rh)

    return FitResult(model, input, ps_out, resid, w, Σ, (; converged = mcmc_ok, rhat = rh))
end

end
