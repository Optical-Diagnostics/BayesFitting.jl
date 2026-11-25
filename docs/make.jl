using BayesFitting
using Documenter

DocMeta.setdocmeta!(BayesFitting, :DocTestSetup, :(using BayesFitting); recursive=true)

makedocs(;
    modules=[BayesFitting],
    authors="Jan Kuhfeld <jankuhfeld@plasma-matters.nl> and contributors",
    sitename="BayesFitting.jl",
    format=Documenter.HTML(;
        canonical="https://optical-diagnostics.github.io/BayesFitting.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Optical-Diagnostics/BayesFitting.jl",
    devbranch="main",
)
