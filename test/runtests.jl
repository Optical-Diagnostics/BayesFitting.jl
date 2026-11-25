using BayesFitting
using Test
using Aqua
using JET
using SafeTestsets

@testset "BayesFitting.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(BayesFitting)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(BayesFitting; target_modules = (BayesFitting,))
    end
end

@safetestset "Fitting methods" begin
    include("test_fitting.jl")
end
