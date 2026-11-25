using Test
using BayesFitting
using NonlinearSolve, Turing
using ComponentArrays
using Distributions
using Statistics
using Random
using ForwardDiff
using AbstractGPs
using LinearAlgebra: diag
Random.seed!(42)

gaussian(x, σ) = exp(-x^2 / (2σ^2)) / (σ * sqrt(2π))
model(xs, p) = @. p.ampl * gaussian(xs, 1.0) + p.bg

xs = range(-5, 5, length = 1000)
σ_noise = 0.1
truth = model(xs, (ampl = 1.0, bg = 1.0))
data = Normal.(truth .+ rand(Normal(0, σ_noise), length(xs)), σ_noise)
raw = mean.(data)

# Exact posterior of a model linear in its parameters, y ≈ Xβ. With known σ
# and Normal priors: mean (XᵀX/σ² + Λ)⁻¹(Xᵀy/σ² + Λμ), covariance
# (XᵀX/σ² + Λ)⁻¹. Without σ: ordinary least squares, with σ̂² = rss/(n - p).
function linear_posterior(X, y; σ = nothing, priors = Normal[])
    if σ === nothing
        A = X' * X
        β = A \ (X' * y)
        return β, sum(abs2, y - X * β) / (size(X, 1) - size(X, 2)) * inv(A)
    end
    A = X' * X / σ^2
    b = X' * y / σ^2
    for (i, d) in enumerate(priors)
        A[i, i] += 1 / d.σ^2
        b[i] += d.μ / d.σ^2
    end
    return A \ b, inv(A)
end

g1 = gaussian.(xs, 1.0)
X = [g1 ones(length(xs))]

nls_ext = Base.get_extension(BayesFitting, :BayesFittingNonlinearSolveExt)
turing_ext = Base.get_extension(BayesFitting, :BayesFittingTuringExt)

priors = ComponentVector(ampl = Normal(1, 1), bg = Normal(1, 1))
plain_params = ComponentVector(ampl = 1.0, bg = 1.0)
fixed_params = ComponentVector(ampl = Normal(1, 1), bg = Dirac(1.0))

lmres = fit_model(model, xs, data, priors, LevenbergMarquardtFit())
turingres = fit_model(model, xs, data, priors, TuringFit(); num_chains = 5, progress = false)
β_lm, Σ_lm = linear_posterior(X, raw; σ = σ_noise, priors = collect(priors))

@testset "least squares: exact linear posterior" begin
    @test collect(coef(lmres)) ≈ β_lm rtol = 1.0e-8
    @test vcov(lmres) ≈ Σ_lm rtol = 1.0e-8
    @test cor(lmres)[1, 2] ≈ Σ_lm[1, 2] / sqrt(Σ_lm[1, 1] * Σ_lm[2, 2]) rtol = 1.0e-8
end

@testset "MCMC agrees with least squares" begin
    # The model is linear, so LM's Laplace approximation is the exact posterior.
    @test collect(coef(turingres)) ≈ collect(coef(lmres)) rtol = 1.0e-2
    @test collect(stderror(turingres)) ≈ collect(stderror(lmres)) rtol = 5.0e-2
    @test cor(turingres)[1, 2] ≈ cor(lmres)[1, 2] atol = 0.1
end

@testset "residual bookkeeping" begin
    # Prior rows belong to the least-squares formulation, not the observations.
    @test nobs(lmres) == length(xs)
    # StatsAPI semantics: dof = consumed (parameters, + σ̂ when estimated),
    # dof_residual = nobs - parameters.
    @test dof_residual(lmres) == length(xs) - 2
    @test dof(lmres) == 2
    @test weights(lmres) ≈ fill(1 / σ_noise, length(xs))
    # Residuals are σ-normalised when the uncertainty is known, so this is a
    # reduced χ² and should sit near 1 for a well-specified model.
    @test reduced_chisq(lmres) ≈ 1.0 atol = 0.15
    @test residuals(lmres) ≈ (raw .- model(xs, coef(lmres))) ./ σ_noise
    @test residuals(turingres) ≈ (raw .- model(xs, coef(turingres))) ./ σ_noise
end

# Known σ with plain point-estimate `params`: no priors, so weighted least
# squares, and the residual carries data rows only.
plain_res = fit_model(model, xs, data, plain_params, LevenbergMarquardtFit())
β_ls, Σ_ls = linear_posterior(X, raw; σ = σ_noise)

@testset "known σ, no priors: weighted least squares" begin
    @test collect(coef(plain_res)) ≈ β_ls rtol = 1.0e-8
    @test collect(stderror(plain_res)) ≈ sqrt.(diag(Σ_ls)) rtol = 1.0e-8
end

# No uncertainty on the data at all: the noise scale has to be estimated from
# the fit, and `mse` is then σ̂² rather than a reduced χ².
noσ_res = fit_model(model, xs, raw, plain_params, LevenbergMarquardtFit())

@testset "unknown σ: ordinary least squares" begin
    β_ols, Σ_ols = linear_posterior(X, raw)
    @test collect(coef(noσ_res)) ≈ β_ols rtol = 1.0e-8
    @test collect(stderror(noσ_res)) ≈ sqrt.(diag(Σ_ols)) rtol = 1.0e-6
    @test mse(noσ_res) ≈ rss(noσ_res) / (length(xs) - 2)
    @test_throws ErrorException reduced_chisq(noσ_res)
end

# Plain-number parameters are scaled by their magnitude, so a negative start
# value must still give a positive standard error.
@testset "negative plain-number start value" begin
    neg = fit_model(model, xs, data, ComponentVector(ampl = 1.0, bg = -1.0), LevenbergMarquardtFit())
    @test collect(coef(neg)) ≈ β_ls rtol = 1.0e-8
    @test collect(stderror(neg)) ≈ sqrt.(diag(Σ_ls)) rtol = 1.0e-8
    neg_ols = fit_model(model, xs, raw, ComponentVector(ampl = 1.0, bg = -1.0), LevenbergMarquardtFit())
    @test all(>(0), stderror(neg_ols))
end

fixed_res = fit_model(model, xs, data, fixed_params, LevenbergMarquardtFit())

@testset "Dirac prior pins a parameter" begin
    @test mean(fixed_res[:bg]) == 1.0
    @test fixed_res.diagnostics.covariance_ok
    # The free parameter is the one-parameter posterior given bg = 1.
    β1, Σ1 = linear_posterior(X[:, 1:1], raw .- 1.0; σ = σ_noise, priors = [Normal(1, 1)])
    @test mean(fixed_res[:ampl]) ≈ β1[1] rtol = 1.0e-8
    @test std(fixed_res[:ampl]) ≈ sqrt(Σ1[1, 1]) rtol = 1.0e-8
    # Zero variance, zero correlation row (diagonal included).
    @test vcov(fixed_res)[2, 2] == 0.0
    @test cor(fixed_res)[2, :] == [0.0, 0.0]
end

@testset "ill-conditioned and saddle covariance" begin
    # Two parameters seen only as a sum: `a - b` is free, so neither error is
    # bounded and both must come out large.
    mab(x, p) = fill(p.a + p.b, length(x))
    dab = Normal.(fill(2.0, 100), 0.1)
    for approx in (false, true)
        rab = @test_logs (:warn, r"ill-conditioned") match_mode = :any fit_model(
            mab, 1:100, dab, ComponentVector(a = 1.0, b = 1.0), LevenbergMarquardtFit();
            approximate_uncertainties = approx)
        @test !rab.diagnostics.covariance_ok
        @test std(rab[:a]) > 1.0e3 && std(rab[:b]) > 1.0e3
        @test mean(rab[:a]) + mean(rab[:b]) ≈ 2.0
    end

    # A saddle with every variance positive: 0.5‖r‖² has Hessian
    # 0.01 I - (1.01/3) ones(3, 3) at 0, eigenvalues (-1, 0.01, 0.01), and zero
    # gradient there. The inverse has all diagonals +66.3, so only the eigenvalue
    # sign shows the saddle. JᵀJ = 0.01 I cannot see it.
    csad = sqrt(1.01 / 3)
    msad(x, p) = (s = csad * (p.x + p.y + p.z); [0.5 * s^2, 0.1 * p.x, 0.1 * p.y, 0.1 * p.z])
    dsad = Normal.([1.0, 0.0, 0.0, 0.0], 1.0)
    psad = ComponentVector(x = 0.0, y = 0.0, z = 0.0)
    rsad = @test_logs (:warn, r"saddle") match_mode = :any fit_model(
        msad, 1:4, dsad, psad, LevenbergMarquardtFit())
    @test !rsad.diagnostics.covariance_ok
    @test fit_model(msad, 1:4, dsad, psad, LevenbergMarquardtFit();
        approximate_uncertainties = true).diagnostics.covariance_ok
end

@testset "Uniform prior" begin
    @test ForwardDiff.derivative(x -> nls_ext.prior_residual(Uniform(0.0, 1.0), x), 5.0) > 0
    @test ForwardDiff.derivative(x -> nls_ext.prior_residual(Uniform(0.0, 1.0), x), -5.0) > 0

    # Inside the box the prior adds nothing: weighted least squares.
    box_params = ComponentVector(ampl = Uniform(0.0, 3.0), bg = Uniform(0.0, 3.0))
    box_res = fit_model(model, xs, data, box_params, LevenbergMarquardtFit())
    @test collect(coef(box_res)) ≈ β_ls rtol = 1.0e-8
    @test collect(stderror(box_res)) ≈ sqrt.(diag(Σ_ls)) rtol = 1.0e-6
    @test isempty(box_res.diagnostics.at_bound)

    wall_params = ComponentVector(ampl = Uniform(0.0, 0.8), bg = Uniform(0.0, 3.0))
    wall_res = @test_logs (:warn, r"at a `Uniform` bound") match_mode = :any fit_model(
        model, xs, data, wall_params, LevenbergMarquardtFit())
    @test wall_res.diagnostics.at_bound == ["ampl"]
    @test mean(wall_res[:ampl]) ≈ 0.8 atol = 1.0e-2

    raw_res = @test_logs (:warn, r"at a `Uniform` bound") fit_model(
        model, xs, raw, wall_params, LevenbergMarquardtFit())
    @test raw_res.diagnostics.at_bound == ["ampl"]
    @test mean(raw_res[:ampl]) ≈ 0.8 atol = 1.0e-2
    @test nobs(raw_res) == length(xs)
    raw_box = fit_model(model, xs, raw, box_params, LevenbergMarquardtFit())
    raw_free = fit_model(model, xs, raw, ComponentVector(ampl = 1.5, bg = 1.5), LevenbergMarquardtFit())
    @test collect(coef(raw_box)) ≈ collect(coef(raw_free)) rtol = 1.0e-6
    @test collect(stderror(raw_box)) ≈ collect(stderror(raw_free)) rtol = 1.0e-6
end

@testset "model output length" begin
    # The residual loop reads the prediction under `@inbounds`.
    short(xs, p) = model(xs, p)[1:(end - 1)]
    long(xs, p) = vcat(model(xs, p), 0.0)
    for m in (short, long), d in (data, raw), fuse in (true, false)
        @test_throws DimensionMismatch fit_model(m, xs, d, ComponentVector(ampl = 1.0, bg = 1.0),
            LevenbergMarquardtFit(); fuse)
    end
end

bgmodel(xs, p) = @. p.ampl * gaussian(xs, 1.0) + p.background[1] + p.background[2] * xs
bgtruth = bgmodel(xs, ComponentVector(ampl = 1.0, background = [1.0, 0.05]))
bgdata = Normal.(bgtruth .+ rand(Normal(0, σ_noise), length(xs)), σ_noise)
bgpriors = ComponentVector(ampl = Normal(1, 1), background = [Normal(0, 1) for _ in 1:2])
bgres = fit_model(bgmodel, xs, bgdata, bgpriors, LevenbergMarquardtFit())

@testset "vector-valued priors" begin
    βbg, Σbg = linear_posterior([g1 ones(length(xs)) xs], mean.(bgdata);
                                σ = σ_noise, priors = collect(bgpriors))
    @test collect(coef(bgres)) ≈ βbg rtol = 1.0e-8
    @test collect(stderror(bgres)) ≈ sqrt.(diag(Σbg)) rtol = 1.0e-8
end

@testset "invalid uncertainties" begin
    @test_throws ErrorException fit_model(model, xs, Normal.(truth, 0.0), priors, LevenbergMarquardtFit())
    @test_throws ErrorException fit_model(model, xs, vcat(Any[data[1]], Any[raw[i] for i in 2:length(raw)]),
                                          priors, LevenbergMarquardtFit())
end

# Fused value+Jacobian: one ForwardDiff pass per solver call, so Float64
# model evaluations reduce to the single buffer-sizing probe.
nprimal = Ref(0)
counting_model(x, p) = begin
    eltype(p) <: ForwardDiff.Dual || (nprimal[] += 1)
    model(x, p)
end

fusedres = fit_model(counting_model, xs, data, priors, LevenbergMarquardtFit())
primal_fused = nprimal[]
nprimal[] = 0
unfusedres = fit_model(counting_model, xs, data, priors, LevenbergMarquardtFit(); fuse = false)
primal_unfused = nprimal[]

@testset "fused value and Jacobian" begin
    @test collect(coef(fusedres)) ≈ collect(coef(unfusedres)) rtol = 1.0e-8
    @test collect(stderror(fusedres)) ≈ collect(stderror(unfusedres)) rtol = 1.0e-6
    @test primal_fused <= 2
    @test primal_unfused > primal_fused + 5   # unfused pays one primal per iteration
end

@testset "solver knobs" begin
    trres = fit_model(model, xs, data, priors, LevenbergMarquardtFit();
        alg = TrustRegion(; autodiff = AutoForwardDiff()))
    @test mean(trres[:ampl]) ≈ mean(fusedres[:ampl]) rtol = 1.0e-6

    dampres = fit_model(model, xs, data, priors, LevenbergMarquardtFit();
        damping_initial = 1.0e-4, damping_decrease_factor = 10.0)
    @test mean(dampres[:ampl]) ≈ mean(fusedres[:ampl]) rtol = 1.0e-6
end

@testset "approximate uncertainties" begin
    # `approximate_uncertainties` swaps the exact Hessian for Gauss-Newton JᵀJ.
    # The test model is linear in its parameters, so the dropped residual-
    # curvature term is zero and the approximation must be exact.
    approxres = fit_model(model, xs, data, priors, LevenbergMarquardtFit();
        approximate_uncertainties = true)
    @test collect(stderror(approxres)) ≈ collect(stderror(fusedres)) rtol = 1.0e-6

    # Nonlinear parameter: the two may differ, but only mildly at a good fit.
    wmodel(x, p) = @. p.ampl * gaussian(x, p.width) + p.bg
    wpriors = ComponentVector(ampl = Normal(1, 1), width = Normal(1, 0.3), bg = Normal(1, 1))
    wexact = fit_model(wmodel, xs, data, wpriors, LevenbergMarquardtFit())
    wapprox = fit_model(wmodel, xs, data, wpriors, LevenbergMarquardtFit();
        approximate_uncertainties = true)
    @test mean(wapprox[:width]) ≈ mean(wexact[:width])
    @test std(wapprox[:width]) ≈ std(wexact[:width]) rtol = 0.1
end

@testset "convergence" begin
    @test converged(lmres)
    @test lmres.diagnostics.covariance_ok
    @test converged(turingres)   # split-R̂ below threshold

    badres = @test_logs (:warn, r"did not converge") match_mode = :any fit_model(
        model, xs, data, priors, LevenbergMarquardtFit(); maxiters = 1)
    @test !converged(badres)

    @test turing_ext.split_rhat(randn(1000, 4)) ≈ 1.0 atol = 0.05
    @test turing_ext.split_rhat(hcat(randn(1000), randn(1000) .+ 5)) > 1.5
    @test isnan(turing_ext.split_rhat(randn(3, 2)))
end

@testset "confint, predict, likelihood" begin
    # LM marginals are Normal, so the 95% interval is mean ± 1.96σ; Turing must agree closely.
    ci = confint(lmres)
    @test ci.lower.ampl ≈ mean(lmres[:ampl]) - 1.959964 * std(lmres[:ampl]) rtol = 1.0e-6
    @test ci.upper.ampl ≈ mean(lmres[:ampl]) + 1.959964 * std(lmres[:ampl]) rtol = 1.0e-6
    cit = confint(turingres)
    @test cit.upper.ampl ≈ ci.upper.ampl atol = 0.2 * std(lmres[:ampl])

    # predict: point prediction, and delta-method bands checked against the
    # closed form [g(x) 1] Σ [g(x) 1]' for the linear model.
    @test predict(lmres) ≈ model(xs, coef(lmres))
    pu = predict(lmres; uncertainty = true)
    σ_manual = @. sqrt(g1^2 * Σ_lm[1, 1] + Σ_lm[2, 2] + 2g1 * Σ_lm[1, 2])
    @test std.(pu) ≈ σ_manual rtol = 1.0e-8

    # Gaussian log-likelihood at the fit; with σ unknown, at σ̂² = rss/n.
    @test loglikelihood(lmres) ≈ sum(logpdf.(Normal.(predict(lmres), σ_noise), raw)) rtol = 1.0e-10
    σ̂ = sqrt(rss(noσ_res) / length(xs))
    @test loglikelihood(noσ_res) ≈ sum(logpdf.(Normal.(predict(noσ_res), σ̂), raw)) rtol = 1.0e-10
    @test aic(lmres) ≈ 2 * 2 - 2 * loglikelihood(lmres)
    @test bic(lmres) ≈ 2 * log(length(xs)) - 2 * loglikelihood(lmres)
    wrongres = fit_model(model, xs, bgdata, priors, LevenbergMarquardtFit())
    @test aic(bgres) < aic(wrongres)
    @test bic(bgres) < bic(wrongres)
    # Unweighted: the estimated noise scale consumes one degree of freedom.
    @test dof(noσ_res) == 3
end

@testset "init for least squares" begin
    init_cv = ComponentVector(ampl = 1.7, bg = 0.4)
    prob_init = nls_ext.create_nonlinear_problem(model, xs, data, priors; init = init_cv)
    @test prob_init.u0 ≈ BayesFitting.scaled(ParameterScaling(priors), init_cv)

    initres = fit_model(model, xs, data, priors, LevenbergMarquardtFit(); init = init_cv)
    @test mean(initres[:ampl]) ≈ mean(lmres[:ampl]) rtol = 1.0e-6

    @test_throws ErrorException fit_model(model, xs, data, priors, LevenbergMarquardtFit();
        init = ComponentVector(ampl = 1.0))

    pinres = fit_model(model, xs, data, fixed_params, LevenbergMarquardtFit();
        init = ComponentVector(ampl = 1.5, bg = 7.0))
    @test mean(pinres[:bg]) == 1.0
end

@testset "show" begin
    @test occursin("ampl = ", sprint(show, lmres)) && occursin("±", sprint(show, lmres))
    @test occursin("1000 observations, converged", sprint(show, MIME("text/plain"), lmres))
    @test occursin("(fixed)", sprint(show, fixed_res))
    @test occursin("background[2]", sprint(show, bgres))
end

@testset "Turing warm start" begin
    warmres = fit_model(model, xs, data, priors, TuringFit();
        init = lmres, num_chains = 2, num_samples = 500, progress = false)
    @test mean(warmres[:ampl]) ≈ mean(lmres[:ampl]) rtol = 2.0e-2
    @test converged(warmres)
    @test_throws ErrorException fit_model(model, xs, data, priors, TuringFit();
        init = ComponentVector(ampl = 1.0), num_samples = 100, progress = false)
end

@testset "matrix-valued priors" begin
    # One column per dataset in the multi-dataset design. Every element must
    # give a prior row.
    matpriors = ComponentVector(ampl = Normal(1, 1), bg = [Normal(0, 1) for c in 1:2, i in 1:2])
    matmodel(x, θ) = @. θ.ampl * gaussian(x, 1.0) +
        θ.bg[1, 1] + θ.bg[2, 1] * x + θ.bg[1, 2] * x^2 + θ.bg[2, 2] * x^3
    mattruth = matmodel(xs, ComponentVector(ampl = 1.0, bg = [1.0 0.02; 0.05 0.001]))
    matdata = Normal.(mattruth .+ rand(Normal(0, σ_noise), length(xs)), σ_noise)
    matres = fit_model(matmodel, xs, matdata, matpriors, LevenbergMarquardtFit())
    # Flat order is column-major: ampl, bg[1,1], bg[2,1], bg[1,2], bg[2,2].
    βm, Σm = linear_posterior([g1 ones(length(xs)) xs xs .^ 2 xs .^ 3], mean.(matdata);
                              σ = σ_noise, priors = collect(matpriors))
    @test collect(coef(matres)) ≈ βm rtol = 1.0e-8
    @test collect(stderror(matres)) ≈ sqrt.(diag(Σm)) rtol = 1.0e-8
    @test occursin("bg[2,1]", sprint(show, matres))
end

@testset "multi-dataset fitting" begin
    # Shared width, private amplitudes and background columns, one global covariance.
    mdmodels = ntuple(i -> (x, θ) ->
        (@. θ.ampl[i] * gaussian(x, θ.width) + θ.mbg[1, i] + θ.mbg[2, i] * x), 2)
    x1 = collect(range(-5, 5, length = 800))
    x2 = collect(range(-4, 4, length = 600))
    mdtruth = ComponentVector(width = 1.0, ampl = [1.0, 0.6], mbg = [0.5 1.0; 0.02 -0.03])
    d1 = Normal.(mdmodels[1](x1, mdtruth) .+ rand(Normal(0, σ_noise), length(x1)), σ_noise)
    d2 = Normal.(mdmodels[2](x2, mdtruth) .+ rand(Normal(0, σ_noise), length(x2)), σ_noise)
    mdparams = ComponentVector(width = Normal(1, 0.3),
        ampl = [Normal(1, 1), Normal(1, 1)], mbg = [Normal(0, 1) for c in 1:2, i in 1:2])

    mdres = fit_model([(mdmodels[1], x1, d1), (mdmodels[2], x2, d2)], mdparams,
        LevenbergMarquardtFit())
    @test converged(mdres)
    @test nobs(mdres) == length(x1) + length(x2)

    s1model(x, θ) = @. θ.ampl * gaussian(x, θ.width) + θ.bg[1] + θ.bg[2] * x
    s1params = ComponentVector(width = Normal(1, 0.3), ampl = Normal(1, 1),
        bg = [Normal(0, 1), Normal(0, 1)])
    s1res = fit_model(s1model, x1, d1, s1params, LevenbergMarquardtFit())
    @test std(mdres[:width]) < std(s1res[:width])

    handmodel(xt, θ) = vcat(mdmodels[1](xt[1], θ), mdmodels[2](xt[2], θ))
    handres = fit_model(handmodel, (x1, x2), vcat(d1, d2), mdparams, LevenbergMarquardtFit())
    @test mean(mdres[:width]) ≈ mean(handres[:width]) rtol = 1.0e-10
    @test std(mdres[:width]) ≈ std(handres[:width]) rtol = 1.0e-8

    mdunfused = fit_model([(mdmodels[1], x1, d1), (mdmodels[2], x2, d2)], mdparams,
        LevenbergMarquardtFit(); fuse = false)
    @test mean(mdunfused[:width]) ≈ mean(mdres[:width]) rtol = 1.0e-8

    # Privates of different datasets talk only through the shared width, so their
    # correlation is weaker than shared-private.
    Cmd = cor(mdres)
    iw, ia1, ia2 = 1, 2, 3   # flat order: width, ampl[1], ampl[2], ...
    @test abs(Cmd[ia1, ia2]) < abs(Cmd[iw, ia1])

    @test length(residuals(mdres; dataset = 2)) == length(x2)
    @test rss(mdres; dataset = 1) + rss(mdres; dataset = 2) ≈ rss(mdres)
    @test nobs(mdres; dataset = 2) == length(x2)
    @test predict(mdres; dataset = 2) ≈ mdmodels[2](x2, coef(mdres))
    hand_band = std.(predict(handres; uncertainty = true))
    @test std.(predict(mdres; dataset = 1, uncertainty = true)) ≈ hand_band[1:length(x1)] rtol = 1.0e-6
    @test occursin("2 datasets", sprint(show, MIME("text/plain"), mdres))

    @test_throws ErrorException fit_model([(mdmodels[1], x1, d1), (mdmodels[2], x2, mean.(d2))],
        mdparams, LevenbergMarquardtFit())
    @test_throws ErrorException fit_model([((x, θ) -> zeros(10), x1, d1)], mdparams,
        LevenbergMarquardtFit())
    @test_throws ErrorException residuals(lmres; dataset = 1)
    @test_throws ErrorException residuals(mdres; dataset = 3)

    mdturing = fit_model([(mdmodels[1], x1, d1), (mdmodels[2], x2, d2)], mdparams,
        TuringFit(); init = mdres, num_chains = 2, num_samples = 400, progress = false)
    @test mean(mdturing[:width]) ≈ mean(mdres[:width]) rtol = 2.0e-2
    @test std(mdturing[:width]) ≈ std(mdres[:width]) rtol = 0.2

    nestfr = FitResult(nothing, nothing,
        ComponentVector(bg = ComponentVector(d1 = [Normal(1.0, 0.1)])), zeros(2), nothing)
    @test occursin("bg.d1[1]", sprint(show, nestfr))
end

# `@testset` restores the global RNG on exit, so seed here to keep the series
# data independent of the testsets above.
Random.seed!(42)
smodel(x, θ) = @. θ.ampl * gaussian(x, θ.width) + θ.bg
ts = collect(1.0:30.0)
sxs = collect(range(-5, 5, length = 60))
width_true(t) = 1.0 + 0.25 * sin(2π * t / 25)
ampl_true(t) = 1.0 + 0.05 * t          # drifts to 2.5, a full prior σ away
sdatas = [Normal.(smodel(sxs, (ampl = ampl_true(t), width = width_true(t), bg = 1.0)) .+
                  0.3 .* randn(length(sxs)), 0.3) for t in ts]
spriors = ComponentVector(ampl = Normal(1, 1.5), width = Normal(1, 0.5), bg = Normal(1, 0.5))
sdrift = GaussianDrift(lengthscale = 6.0)

sres = fit_series(smodel, ts, sxs, sdatas, spriors, LevenbergMarquardtFit();
    drift = sdrift, track = NotTracked(:bg))

struct NoDrift <: DriftModel end

@testset "series with GP drift" begin
    @test length(sres) == 30
    @test all(converged(sres))
    @test coef(sres)[3, :] ≈ collect(coef(sres[3]))
    tr = trajectory(sres, :width)
    @test maximum(abs.(tr.mean .- width_true.(ts)) ./ tr.std) < 5
    # LM frames have Normal marginals: the interval is mean ± 1.96σ.
    @test tr.upper .- tr.mean ≈ 1.959964 .* tr.std rtol = 1.0e-6
    @test tr.mean .- tr.lower ≈ 1.959964 .* tr.std rtol = 1.0e-6

    # The GP predictive carries information between frames, so the drifting
    # amplitude is tracked far from the prior mean and each frame's reported σ is
    # smaller than an independent fit's.
    tra = trajectory(sres, :ampl)
    @test maximum(abs.(tra.mean .- ampl_true.(ts)) ./ tra.std) < 5
    indep_σ = [std(fit_model(smodel, sxs, d, spriors, LevenbergMarquardtFit())[:ampl])
               for d in sdatas]
    @test mean(tra.std) < 0.95 * mean(indep_σ)

    @test_throws ErrorException fit_series(smodel, ts[1:2], sxs, sdatas[1:2], spriors,
        LevenbergMarquardtFit(); drift = sdrift, track = Tracked(:nope))
    pinpriors = ComponentVector(ampl = Normal(1, 0.5), width = Dirac(1.0), bg = Normal(1, 0.5))
    @test_throws ErrorException fit_series(smodel, ts[1:2], sxs, sdatas[1:2], pinpriors,
        LevenbergMarquardtFit(); drift = sdrift, track = Tracked(:width))
    @test sort(BayesFitting._tracked_keys(Tracked(:ampl), spriors)) ==
          sort(setdiff(BayesFitting._tracked_keys(nothing, spriors),
                       BayesFitting._tracked_keys(NotTracked(:ampl), spriors)))
end

@testset "GP drift against the closed form" begin
    μ0, σ0, ℓ = 2.0, 0.5, 2.0
    gd = GaussianDrift(lengthscale = ℓ, kernel = with_lengthscale(SqExponentialKernel(), ℓ))
    gstate = BayesFitting.init_drift(gd, [:a], ComponentVector(a = Normal(μ0, σ0)))
    t1, y1, σobs = 1.0, 2.4, 0.1
    BayesFitting.drift_update!(gstate, :a, t1, y1, σobs)
    t2 = 2.0
    kk = exp(-(t2 - t1)^2 / (2ℓ^2))
    n1 = (gd.noise_inflation * σobs)^2
    μref = μ0 + σ0^2 * kk * (y1 - μ0) / (σ0^2 + n1)
    σref = sqrt(σ0^2 * (1 - σ0^2 * kk^2 / (σ0^2 + n1)))
    μp, σp = BayesFitting.drift_predict(gstate, :a, t2)
    @test μp ≈ μref rtol = 1.0e-10
    @test σp ≈ σref rtol = 1.0e-10

    gd1 = GaussianDrift(lengthscale = ℓ, kernel = with_lengthscale(SqExponentialKernel(), ℓ),
        window = 1)
    g1s = BayesFitting.init_drift(gd1, [:a], ComponentVector(a = Normal(μ0, σ0)))
    BayesFitting.drift_update!(g1s, :a, 0.0, 5.0, σobs)   # to be forgotten
    BayesFitting.drift_update!(g1s, :a, t1, y1, σobs)
    μw, σw = BayesFitting.drift_predict(g1s, :a, t2)
    @test μw ≈ μref rtol = 1.0e-10
    @test σw ≈ σref rtol = 1.0e-10
end

@testset "series failure handling and MCMC" begin
    baddatas = [copy(d) for d in sdatas[1:6]]
    baddatas[3] = Normal.(fill(NaN, length(sxs)), 0.05)
    bres = @test_logs (:warn, r"frame 3") match_mode = :any fit_series(
        smodel, ts[1:6], sxs, baddatas, spriors, LevenbergMarquardtFit();
        drift = GaussianDrift(lengthscale = 6.0), track = NotTracked(:bg))
    @test converged(bres)[3] == false
    @test count(converged(bres)) >= 4
    @test all(isfinite, trajectory(bres, :width).mean)

    @test_throws ErrorException fit_series(smodel, ts[1:2], sxs, sdatas[1:2], spriors,
        LevenbergMarquardtFit(); drift = NoDrift())

    # 600 samples: at 300 a single chain's split-R̂ sits at the 1.05 threshold.
    tres = fit_series(smodel, ts[1:3], sxs, sdatas[1:3], spriors, TuringFit();
        drift = GaussianDrift(lengthscale = 6.0), track = NotTracked(:bg),
        num_samples = 600, progress = false)
    @test all(converged(tres))
    @test all(abs.(trajectory(tres, :width).mean .- trajectory(sres, :width).mean[1:3]) .<
              4 .* trajectory(sres, :width).std[1:3])
end

@testset "StaticDrift" begin
    stres = fit_series(smodel, ts[1:5], sxs, sdatas[1:5], spriors, LevenbergMarquardtFit())
    indep5 = fit_model(smodel, sxs, sdatas[5], spriors, LevenbergMarquardtFit())
    @test mean(stres[5][:ampl]) ≈ mean(indep5[:ampl]) rtol = 1.0e-6
    @test std(stres[5][:ampl]) ≈ std(indep5[:ampl]) rtol = 1.0e-4

    boxpriors = ComponentVector(ampl = Uniform(0.0, 5.0), width = Normal(1, 0.5), bg = Normal(1, 0.5))
    ststate = BayesFitting.init_drift(StaticDrift(), [:ampl, :width], boxpriors)
    @test BayesFitting.drift_prior(ststate, :ampl, 1.0, boxpriors.ampl) === boxpriors.ampl
    stbox = fit_series(smodel, ts[1:2], sxs, sdatas[1:2], boxpriors, LevenbergMarquardtFit())
    indep2 = fit_model(smodel, sxs, sdatas[2], boxpriors, LevenbergMarquardtFit())
    @test collect(coef(stbox[2])) ≈ collect(coef(indep2)) rtol = 1.0e-6
end

@testset "parameter scaling" begin
    sc_priors = ComponentVector(a = Normal(2.0, 0.5), b = Dirac(1.0),
                                c = [Normal(0.0, 1e-3), Normal(0.0, 1e-3)])
    ps = ParameterScaling(sc_priors)
    θ = ComponentVector(a = 3.0, b = 1.0, c = [0.01, -0.02])
    z = BayesFitting.scaled(ps, θ)
    @test z ≈ [2.0, 0.0, 10.0, -20.0]
    @test BayesFitting.unscaled(ps, z) ≈ collect(θ)
    @test BayesFitting.unscaled(ps, [5.0, 7.0, 0.0, 0.0])[2] == 1.0
    @test BayesFitting.unscaled_cov(ps, [1.0 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]) ≈
          [0.25 0 0 0; 0 0 0 0; 0 0 1e-6 0; 0 0 0 1e-6]

    ps2 = ParameterScaling(sc_priors; scale = (; a = 10.0, c = [1.0, 1.0]))
    @test ps2.σ == [10.0, 0.0, 1.0, 1.0]
    @test_throws ErrorException ParameterScaling(sc_priors; scale = (; b = 1.0))
    @test_throws ErrorException ParameterScaling(sc_priors; scale = (; a = -1.0))

    scaled_res = fit_model(model, xs, data, priors, LevenbergMarquardtFit(); scale = (; ampl = 100.0))
    @test coef(scaled_res).ampl ≈ coef(lmres).ampl rtol = 1e-6
    @test stderror(scaled_res).ampl ≈ stderror(lmres).ampl rtol = 1e-4
end

# Parallel chains get their own copy of the model, so a model with a mutable
# cache is safe to sample with several chains.
struct CachedModel
    cache::Dict{DataType, Vector}
end
function (m::CachedModel)(xs, p)
    T = typeof(p.ampl * gaussian(xs[1], 1.0) + p.bg)
    buf = get!(m.cache, T) do
        Vector{T}(undef, length(xs))
    end
    @. buf = p.ampl * gaussian(xs, 1.0) + p.bg
    return copy(buf)
end

@testset "MCMC scaling, pinning and model copies" begin
    # A badly scaled parameter: the amplitude in units 1e10 larger. The sampler
    # works in scaled units, so this must give the least-squares posterior.
    big_model(xs, p) = @. p.ampl * 1e-10 * gaussian(xs, 1.0) + p.bg
    big_priors = ComponentVector(ampl = Normal(1e10, 1e10), bg = Normal(1, 1))
    big_lm = fit_model(big_model, xs, data, big_priors, LevenbergMarquardtFit())
    big_mc = fit_model(big_model, xs, data, big_priors, TuringFit(); num_samples = 300, progress = false)
    @test converged(big_mc)
    @test mean(big_mc[:ampl]) ≈ mean(big_lm[:ampl]) rtol = 1e-2
    @test std(big_mc[:ampl]) ≈ std(big_lm[:ampl]) rtol = 0.3

    cached = CachedModel(Dict{DataType, Vector}())
    pin_mc = fit_model(cached, xs, data, fixed_params, TuringFit(); init = ComponentVector(ampl = 1.0, bg = 1.0),
        num_samples = 200, num_chains = 4, progress = false)
    @test converged(pin_mc)
    @test mean(pin_mc[:bg]) == 1.0
    @test vcov(pin_mc)[2, 2] == 0.0
    @test mean(pin_mc[:ampl]) ≈ mean(fixed_res[:ampl]) rtol = 2e-2
end

@testset "pinned parameters and degrees of freedom" begin
    # `fixed_params` fits `ampl` only.
    @test dof_residual(fixed_res) == length(xs) - 1
    @test dof(fixed_res) == 1
    # Unweighted: σ̂² divides by n minus the free parameters.
    noσ_fixed = @test_logs (:warn, r"priors on the parameters are ignored") fit_model(
        model, xs, raw, fixed_params, LevenbergMarquardtFit())
    @test dof(noσ_fixed) == 2
    @test mse(noσ_fixed) ≈ rss(noσ_fixed) / (length(xs) - 1)

    zres = fit_model(model, xs, data, ComponentVector(ampl = Normal(1, 1), bg = Normal(1.0, 0.0)),
        LevenbergMarquardtFit())
    @test mean(zres[:bg]) == 1.0
    @test mean(zres[:ampl]) ≈ mean(fixed_res[:ampl])

    @test_throws ArgumentError fit_model(model, xs, raw, priors, TuringFit(); progress = false)
end

@testset "code errors are rethrown" begin
    broken(x, p) = undefined_function_in_model(x, p)
    @test_throws UndefVarError fit_series(broken, ts[1:3], sxs, sdatas[1:3], spriors,
        LevenbergMarquardtFit())
    # MCMC chains run in tasks; the wrapped error is unwrapped before deciding.
    failed = try
        wait(Threads.@spawn undefined_function_in_model())
    catch e
        e
    end
    @test BayesFitting._code_error(failed)
    @test BayesFitting._code_error(CompositeException([failed]))
    @test !BayesFitting._code_error(ArgumentError("bad frame"))
end

@testset "init from a FitResult; GP noise floor" begin
    warm = fit_model(model, xs, data, priors, LevenbergMarquardtFit(); init = lmres)
    @test coef(warm) ≈ coef(lmres) rtol = 1e-6
    @test_throws ErrorException fit_model(model, xs, data, priors, LevenbergMarquardtFit();
        init = sres[1])

    # The GP drift's noise floor scales with the prior variance. A zero-σ
    # observation of a parameter of size 1e-8 must pin the prediction there.
    gs = BayesFitting.init_drift(GaussianDrift(lengthscale = 5.0), [:a],
        ComponentVector(a = Normal(1.0e-8, 1.0e-9)))
    BayesFitting.drift_update!(gs, :a, 0.0, 1.2e-8, 0.0)
    μs, σs = BayesFitting.drift_predict(gs, :a, 0.0)
    @test μs ≈ 1.2e-8 rtol = 1e-3
    @test σs < 1.0e-12
end

@testset "Uniform bounds hold against informative data" begin
    # 1000 points at σ = 1e-3 pull k to 5; the box ends at 1. The estimate is
    # the bound itself.
    line(x, θ) = @. θ.k * x
    lx = range(0, 1, length = 1000)
    ly = 5.0 .* lx .+ 1.0e-3 .* randn(1000)
    for σd in (1.0e-1, 1.0e-3, 1.0e-5)
        r = @test_logs (:warn, r"at a `Uniform` bound") match_mode = :any fit_model(
            line, lx, Normal.(ly, σd), ComponentVector(k = Uniform(0, 1)), LevenbergMarquardtFit())
        @test coef(r).k == 1.0
        @test r.diagnostics.at_bound == ["k"]
        # The uncertainty is that of the unbounded problem, not of the wall.
        @test stderror(r).k ≈ σd / sqrt(sum(abs2, lx)) rtol = 1.0e-6
    end

    # The model is never evaluated outside the box: `sqrt` of a parameter
    # bounded at zero, with the data pulling it below. Its derivative is
    # infinite at the bound, so there is no covariance, and that is reported.
    root(x, θ) = @. sqrt(θ.w) * x
    rr = @test_logs (:warn, r"not finite") match_mode = :any fit_model(
        root, lx, Normal.(1.0e-2 .* randn(1000) .- 0.05 .* lx, 1.0e-2),
        ComponentVector(w = Uniform(0, 10)), LevenbergMarquardtFit())
    @test coef(rr).w == 0.0
    @test !converged(rr) && !rr.diagnostics.covariance_ok
    @test stderror(rr).w == Inf

    two(x, θ) = @. θ.k * x + θ.c
    d2 = Normal.(5.0 .* lx .+ 0.3 .+ 1.0e-2 .* randn(1000), 1.0e-2)
    p2 = ComponentVector(k = Uniform(0, 1), c = Normal(0, 5))
    lm2 = @test_logs (:warn,) match_mode = :any fit_model(two, lx, d2, p2, LevenbergMarquardtFit())
    @test coef(lm2).k == 1.0
    mc2 = fit_model(two, lx, d2, p2, TuringFit(); init = lm2, num_samples = 200, progress = false)
    @test 0.99 < coef(mc2).k <= 1.0
end

@testset "priors next to plain numbers" begin
    # A plain number makes the element type `Any`; array and nested priors
    # must still be found, one element at a time.
    mixed = ComponentVector(a = 1.0, b = [Normal(0, 1.0e-6), Normal(0, 1.0e-6)],
        c = (x = Uniform(0, 1), y = 2.0))
    dists, index = BayesFitting.flatten_priors(mixed)
    @test index == [2, 3, 4]
    @test dists[3] == Uniform(0, 1)

    quad(x, θ) = @. θ.a + θ.b[1] * x + θ.b[2] * x^2
    qx = range(0, 1, length = 50)
    qd = Normal.(quad(qx, (a = 1.0, b = [2.0, 3.0])) .+ 0.01 .* randn(50), 0.01)
    tight = ComponentVector(a = 1.0, b = [Normal(0, 1.0e-6), Normal(0, 1.0e-6)])
    rq = fit_model(quad, qx, qd, tight, LevenbergMarquardtFit())
    @test maximum(abs, coef(rq).b) < 1.0e-5
    box = ComponentVector(a = 1.0, b = [Uniform(0, 1), Uniform(0, 1)])
    rb = @test_logs (:warn,) match_mode = :any fit_model(quad, qx, qd, box, LevenbergMarquardtFit())
    @test coef(rb).b == [1.0, 1.0]
    @test rb.diagnostics.at_bound == ["b[1]", "b[2]"]
end

@testset "GP drift counts each frame once" begin
    # The prior divided out of a Gaussian posterior leaves the likelihood.
    prior = Normal(1.0, 2.0)
    yl, σl = 3.0, 0.5
    pp = 1 / σl^2 + 1 / prior.σ^2
    μpost = (yl / σl^2 + prior.μ / prior.σ^2) / pp
    μb, σb = BayesFitting._likelihood_only(μpost, 1 / sqrt(pp), prior)
    @test μb ≈ yl && σb ≈ σl
    @test BayesFitting._likelihood_only(1.3, 2.0, prior)[2] > 1.0e5 * prior.σ

    # Constant truth and a drift far slower than the series: the filter must
    # reproduce the pooled estimate.
    flat(x, θ) = fill(θ.a, length(x))
    nf = 120
    fdatas = [Normal.(1.0 .+ randn(5), 1.0) for _ in 1:nf]
    fres = fit_series(flat, 1:nf, 1:5, fdatas, ComponentVector(a = Normal(0, 10)),
        LevenbergMarquardtFit(); drift = GaussianDrift(lengthscale = 1.0e6, window = nf))
    @test all(converged(fres))
    @test std(fres[nf][:a]) ≈ 1 / sqrt(5nf) rtol = 2.0e-2
    @test mean(fres[nf][:a]) ≈ mean(mean(mean.(d)) for d in fdatas) atol = 0.2 / sqrt(5nf)
end

@testset "GP drift keeps non-Normal priors" begin
    boxpriors = ComponentVector(ampl = Uniform(0.0, 5.0), width = Normal(1, 0.5), bg = Normal(1, 0.5))
    gd = GaussianDrift(lengthscale = 6.0)
    @test BayesFitting._tracked_keys(nothing, boxpriors, gd) == [:width, :bg]
    @test BayesFitting._tracked_keys(NotTracked(:bg), boxpriors, gd) == [:width]
    @test BayesFitting._tracked_keys(nothing, boxpriors, StaticDrift()) == [:ampl, :width, :bg]
    @test_throws ErrorException fit_series(smodel, ts[1:2], sxs, sdatas[1:2], boxpriors,
        LevenbergMarquardtFit(); drift = gd, track = Tracked(:ampl))
    tightbox = ComponentVector(ampl = Uniform(0.0, 0.5), width = Normal(1, 0.5), bg = Normal(1, 0.5))
    bx = @test_logs (:warn,) match_mode = :any fit_series(smodel, ts[1:4], sxs, sdatas[1:4],
        tightbox, LevenbergMarquardtFit(); drift = gd)
    @test all(==(0.5), trajectory(bx, :ampl).mean)
end

@testset "input checks" begin
    lin(x, θ) = @. θ.a + θ.b * x
    # No residual degrees of freedom to estimate the noise from.
    @test_throws ArgumentError fit_model(lin, [0.0, 1.0], [1.0, 2.0],
        ComponentVector(a = 0.5, b = 0.5), LevenbergMarquardtFit())
    # A prior without a finite width cannot set the parameter scale.
    @test_throws ErrorException fit_model(lin, xs, data,
        ComponentVector(a = Normal(0, Inf), b = Normal(0, 1)), LevenbergMarquardtFit())
    # Nothing to sample.
    @test_throws ArgumentError fit_model(lin, xs, data,
        ComponentVector(a = Dirac(1.0), b = Dirac(2.0)), TuringFit(); progress = false)
    # `init` from a FitResult: same length, other layout.
    ab = fit_model(lin, xs, data, ComponentVector(a = Normal(0, 5), b = Normal(0, 5)), LevenbergMarquardtFit())
    @test_throws ErrorException fit_model(lin, xs, data,
        ComponentVector(b = Normal(0, 5), a = Normal(0, 5)), LevenbergMarquardtFit(); init = ab)

    wide(x, θ) = [θ.a .+ θ.b .* x θ.a .- θ.b .* x]
    wx = collect(range(0, 1, length = 20))
    wd = Normal.(vec(wide(wx, (a = 1.0, b = 2.0))) .+ 0.1 .* randn(40), 0.1)
    wr = fit_model(wide, wx, wd, ComponentVector(a = Normal(0, 5), b = Normal(0, 5)), LevenbergMarquardtFit())
    @test size(predict(wr; uncertainty = true)) == (20, 2)

    # A shared input that is a vector of arrays, one per frame by coincidence.
    cat2(x, θ) = θ.a .* vcat(x...)
    gx = [[1.0, 2.0], [3.0, 4.0]]
    gd2 = [Normal.(2.0 .* vcat(gx...) .+ 0.1 .* randn(4), 0.1) for _ in 1:2]
    gr = fit_series(cat2, 1:2, gx, gd2, ComponentVector(a = Normal(1, 5)), LevenbergMarquardtFit();
        per_frame_inputs = false)
    @test all(converged(gr))

    @test BayesFitting._pm_string(12345.678, 123.4) == "12350.0 ± 120.0"
    @test BayesFitting._pm_string(1.23456, 0.0123) == "1.235 ± 0.012"
end
