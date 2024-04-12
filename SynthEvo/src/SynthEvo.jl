module SynthEvo

using DifferentialEquations, RecursiveArrayTools, DiffEqParamEstim, StaticArrays, ModelingToolkit, LinearAlgebra, Statistics 
using Catalyst, Combinatorics, GraphViz, Symbolics, Plots
using JLD2, SimpleDiffEq, LatinHypercubeSampling

abstract type CRN end

include("CRNUtils.jl")

export make_base_problem, run_SF_CPU, runp_SF_CPU, vec2mat, sortAndFilterReactions

include("CRNOptim.jl")

export adagrad_update_get_coefficient, ADAM_update_get_coefficient

include("CRNSymOps.jl")

export sensitivity, make_sensitivity_ode, add_control_equations, sensitivity_from_ode
export symbolic_gradient_descent, joint_jacobian, compute_homeostatic_coefficient

include("SymLosses.jl")

export on_species
export adaptation_loss, sensitivity_loss, ultrasensitivity_loss, steady_state_loss, regularization_loss, weighted_loss
export eval_loss, sensitivity_from_ode, merge_solutions, loss_wrapper

include("NetworkTemplates/FullyConnectedNonExplosive.jl")

# --- todo; maybe it is necessary to rename these functions somewhen ---

export FullyConnectedCRN, reorder2hidden, reorder2visible
export make_FullyConnectedNonExplosive_CRN
export species, control, parameters, number_of_parameters, number_of_parameters, sensitivity_variables, time, n_species
export count_parameters_FullyConnectedNonExplosive, create_reactions
export make_base_problem_for_FCNE_CRN

include("Plotting.jl")

export plot_history, quick_trajectory_plot, quick_sensitivity_plot
export symGD_plot_loss, symGD_plot_parameters, symGD_plot_gradient
export quick_IO_plot

include("CRNevo.jl")

export initialize_state, prepare_GA_loss, symbolic_evolve, symbolic_evolve_NFB

include("Utils.jl")

export save_ga_state, load_ga_state, sample_parameters_from_LHC

end