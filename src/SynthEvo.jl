module SynthEvo

using DifferentialEquations, RecursiveArrayTools, DiffEqParamEstim, StaticArrays
#using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO, OptimizationMOI, Ipopt
#using CSV, DataFrames, 
using Catalyst, Combinatorics, GraphViz, Symbolics, Plots

abstract type CRN end

include("CRNUtils.jl")
include("CRNOptim.jl")
include("CRNSymOps.jl")
include("SymLosses.jl")
include("NetworkTemplates/FullyConnectedNonExplosive.jl")
#include("CRNExplore.jl")
#include("SymbolicOps.jl")
#include("CRNEvo.jl")

end