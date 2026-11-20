using FittingUtils
using Test
using Aqua
using JET
using SafeTestsets

@testset "FittingUtils.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(FittingUtils)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(FittingUtils; target_modules = (FittingUtils,))
    end
end

@safetestset "Fitting methods" begin
    include("test_fitting.jl")
end
