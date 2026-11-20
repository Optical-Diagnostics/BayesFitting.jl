"""
    fit_model(model, input, data, priors, method; kwargs...)

Fit `model(input, θ)` to `data`. `priors` is a `ComponentVector` of
distributions with the layout of the parameters `θ`; their means are the
start values. `data` is a vector of numbers, or of distributions carrying the
per-point uncertainty. Returns a [`FitResult`](@ref).

Every method solves for the parameters in units of their prior width, centred
on the prior mean ([`ParameterScaling`](@ref)). `scale = (; T = 10.0)` overrides
the width of a component, for priors that are deliberately wide. `init` sets a
start point other than the prior means: a `ComponentVector` in the layout of
`priors`, or a previous `FitResult`.

# `LevenbergMarquardtFit()`

Needs `using NonlinearSolve`. Priors may be `Normal`, `Uniform` (a box), or
`Dirac` (pinned). A plain number is a free parameter without a prior, starting
at that value. `Normal` priors take effect only when `data` carries per-point
uncertainty; without it they are ignored, with a warning. Bounds and pins
always hold: the model is only ever evaluated inside the box. A parameter that
ends at a bound is warned about and listed in `diagnostics.at_bound`. The reported
covariance is that of the problem without the box: for a parameter at a bound
it says how well the data determine it there, ignoring the bound, so its
interval can reach outside the box. The model output must have the length of
`data`. Keywords:

- `maxiters = 1000`, `abstol = 1e-6`, `reltol = 1e-3`, `termination_condition`
- `damping_initial = 1.0`, `damping_increase_factor = 2.0`,
  `damping_decrease_factor = 3.0`, `disable_geodesic = Val(true)`,
  `autodiff = AutoForwardDiff()`, or a whole `alg` (e.g. `TrustRegion()`)
- `fuse`: one ForwardDiff pass per residual call yields value and Jacobian;
  on by default with `AutoForwardDiff` and no geodesic acceleration
- `approximate_uncertainties = false`: `true` uses the Gauss-Newton `JᵀJ`
  instead of the exact Hessian for the covariance
- `show_trace = Val(false)`, `trace_level`

# `TuringFit()`

Needs `using Turing`, and `data` with per-point uncertainty. Priors must be
`Normal`, `Uniform`, or `Dirac`. Keywords: `sampler = NUTS()`,
`num_samples = 1000`, `num_chains = 1`, `progress = true`. Chains run in
parallel on `deepcopy`s of the model when Julia has threads. A model with
mutable state must therefore be passed as the object itself, not as a closure
over a global: `deepcopy` does not copy what a closure references.
"""
fit_model

fit_model(model, input, data, priors, method::M; kwargs...) where M <: FitMethod =
    error("fit_model not implemented for method $(typeof(method)). Try `using $(required_package(method))`.")


"""
    inverse_sigmas(data) -> Vector or nothing

Per-point weights `1 / σᵢ` when `data` carries its own uncertainty (each element
a `Distribution`), `nothing` when it does not (plain numbers). Mixed or invalid
(non-positive, non-finite σ) input throws, naming the offending indices.
"""
function inverse_sigmas(data)
    isdist = map(d -> d isa Distributions.Distribution, data)
    any(isdist) || return nothing
    if !all(isdist)
        error("`data` mixes distributions with plain numbers (plain at indices " *
              "$(findall(!, isdist))). Supply an uncertainty for every point, or none.")
    end
    σ = map(Statistics.std, data)
    bad = findall(s -> !(isfinite(s) && s > zero(s)), σ)
    if !isempty(bad)
        error("`data` carries a non-positive or non-finite uncertainty at indices " *
              "$(bad). A point with σ = 0 has infinite weight; drop it or give it a real σ.")
    end
    return inv.(σ)
end
