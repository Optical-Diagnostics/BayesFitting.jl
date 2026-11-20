# `ranges` maps each dataset to its rows in the stacked model output.
struct DatasetStack{M <: Tuple, X <: Tuple, N}
    models::M
    inputs::X
    ranges::NTuple{N, UnitRange{Int}}
end

(ds::DatasetStack)(inputs::Tuple, θ) =
    reduce(vcat, map((m, x) -> m(x, θ), ds.models, inputs))

function _dataset_stack(datasets, params)
    isempty(datasets) && error("`datasets` is empty.")
    all(t -> t isa Tuple && length(t) == 3, datasets) ||
        error("each dataset must be a `(model, input, data)` tuple.")
    models = Tuple(first.(datasets))
    inputs = Tuple(getindex.(datasets, 2))
    all_data = getindex.(datasets, 3)

    unweighted = [inverse_sigmas(d) === nothing for d in all_data]
    if any(unweighted) && !all(unweighted)
        error("datasets mix known and unknown uncertainty (unweighted: dataset(s) " *
              "$(findall(unweighted))). Supply σ for every dataset, or for none.")
    end

    θ0 = map(Statistics.mean, params)
    for (i, (m, x, d)) in enumerate(zip(models, inputs, all_data))
        ny = length(m(x, θ0))
        ny == length(d) || error(
            "dataset $i: model output has length $ny, data has length $(length(d)).")
    end

    stops = cumsum(length.(all_data))
    ranges = ntuple(i -> (i == 1 ? 1 : stops[i - 1] + 1):stops[i], length(stops))
    return DatasetStack(models, inputs, ranges), reduce(vcat, all_data)
end

"""
    fit_model(datasets, params, method; kwargs...)

Fit several datasets simultaneously. `datasets` is a vector of
`(model, input, data)` tuples; every model receives the full global `params`
vector and reads what it needs, so a shared parameter is one read by several
models and a private parameter is a dataset-indexed component (`θ.ampl[i]`, or
a matrix column `θ.bg[:, i]` per dataset).

Every keyword of the chosen method applies. Per-dataset views of the result:
`residuals`/`rss`/`nobs`/`predict` with `dataset = i`.
"""
function fit_model(datasets::AbstractVector{<:Tuple}, params, method::FitMethod; kwargs...)
    stack, data = _dataset_stack(datasets, params)
    return fit_model(stack, stack.inputs, data, params, method; kwargs...)
end

function _dataset_range(fr, i::Integer)
    fr.model isa DatasetStack ||
        error("the `dataset` keyword requires a multi-dataset fit.")
    1 <= i <= length(fr.model.ranges) ||
        error("dataset $i out of range (fit has $(length(fr.model.ranges)) datasets).")
    return fr.model.ranges[i]
end
