module SynthEvo

using DifferentialEquations, RecursiveArrayTools, DiffEqParamEstim, StaticArrays, ModelingToolkit, LinearAlgebra, Statistics 
using Catalyst, Combinatorics, GraphViz, Symbolics, Plots
using JLD2

abstract type CRN end

include("CRNUtils.jl")
include("CRNOptim.jl")
include("CRNSymOps.jl")
include("SymLosses.jl")
include("NetworkTemplates/FullyConnectedNonExplosive.jl")
include("Plotting.jl")
include("CRNevo.jl")

end