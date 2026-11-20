safe_mean(x) = mean(x) |> y -> y == 0 ? 1 : y
# A plain number has `std` NaN: its magnitude is the scale.
safe_std(x) = std(x) |> y -> isnan(y) ? abs(safe_mean(x)) : y

# A zero-width prior pins its parameter. The guard avoids 0/0.
_scaled(p, μ, σ) = iszero(σ) ? zero((p - μ) / oneunit(σ)) : (p - μ) / σ

"""
    ParameterScaling(priors; scale = nothing)

Affine map between physical parameters and the scaled parameters the fitting
methods work with: `z = (θ - μ) / σ`, with `μ` the prior means and `σ` the
prior widths (`std`). A zero-width prior (`Dirac`) pins its parameter: it is
not free, and every scaled value maps back to the prior mean.

`scale` overrides the widths of some components, e.g. `scale = (; T = 10.0)`
or `scale = (; background = fill(1e-3, 7))`, for priors that are deliberately
wide. Widths must be positive, and a pinned parameter cannot be scaled.

Fields: `μ`, `σ` (flat vectors in the order of the priors), `free` (a
`BitVector`, `false` where pinned), `axes` (the component layout).
"""
struct ParameterScaling{V<:AbstractVector, A}
    μ::V
    σ::V
    free::BitVector
    axes::A
end

function ParameterScaling(priors; scale = nothing)
    μc = map(mean, priors)
    σc = map(safe_std, priors)
    free = .!iszero.(getdata(σc))
    if scale !== nothing
        for (k, v) in pairs(scale)
            all(>(0), v) || error("scale for `$k` must be positive, got $v")
            all(!iszero, σc[k]) || error("`$k` is pinned by a zero-width prior and cannot be scaled")
            if σc[k] isa Number
                σc[k] = v
            else
                σc[k] .= v
            end
        end
    end
    μ = float.(getdata(μc))
    σ = float.(getdata(σc))
    bad = findall(i -> !(isfinite(μ[i]) && isfinite(σ[i])), eachindex(μ))
    isempty(bad) || error(
        "priors need a finite mean and width, got otherwise for " *
        "$(ComponentArrays.labels(priors)[bad]). For a parameter without a prior " *
        "pass a plain number (its start value).")
    return ParameterScaling(μ, σ, free, getaxes(priors))
end

function scaled(ps::ParameterScaling, θ)
    getaxes(θ) == ps.axes || error("parameters must have the layout of the priors")
    return _scaled.(getdata(map(mean, θ)), ps.μ, ps.σ)
end

unscaled(ps::ParameterScaling, z) = z .* ps.σ .+ ps.μ

function unscaled_cov(ps::ParameterScaling, Σ)
    D = LinearAlgebra.Diagonal(ps.σ)
    return Matrix(LinearAlgebra.Symmetric(D * Σ * D))
end

# Elements are tested one by one, since a `ComponentVector` mixing numbers and
# distributions has element type `Any`.
function flatten_priors(params)
    flat = getdata(params)
    index = findall(d -> d isa Distributions.Distribution, vec(flat))
    return identity.(Distributions.Distribution[flat[i] for i in index]), index
end
