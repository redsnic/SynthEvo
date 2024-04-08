
# function V(basename, zzz)
#     """
#     Function to create a variable given a basename and a number/suffix

#     Args:
#     - basename: the base name of the variable
#     - zzz: the suffix of the variable

#     Returns:
#     - the variable
#     """
#     return Meta.parse("$(basename)_$(zzz)")
# end

### events and integration

function make_perturbation_event_random_step_function(input, intensity)
    """
    Define a callback function for a pertubation event.

    Args:
    - `input`: the input to the system
    - `intensity`: the intensity of the perturbation

    Returns:
    - the callback function
    """
    function perturb!(integrator, input = input, intensity = intensity)
        integrator[:U] = max(0., input + (rand()*intensity-intensity/2) )
    end
    return perturb!
end


function integrate(prob, tspan, callback, when, algorithm, reltol=1e-8, abstol=1e-8, maxiters=100)
    """
    Integrate the problem with a callback function 
    that will be called only halfway through the time span.

    Args:
    - `prob`: the ODE problem
    - `tspan`: the time span
    - `callback`: the callback function
    - `when`: the time when the callback should be called
    - `algorithm`: the algorithm to use Tsit5() 
    - `reltol`: the relative tolerance
    - `abstol`: the absolute tolerance
    - `maxiters`: the maximum number of iterations

    Returns:
    - the result of the numerical integration
    """
    condition = when #[(tspan[2] - tspan[1])/2]
    ps_cb = PresetTimeCallback(condition, callback)
    sol = solve(prob, algorithm, reltol=reltol, abstol=abstol, callback=ps_cb, maxiters=maxiters)
end

### parameters and variables

# function assemble_opt_parameters_and_varables(p, N)
#     """
#     Assemble the optimization parameters and variables.
#     It will return a 0. initial condition for all the species.

#     Args:
#     - `p`: the parameters
#     - `N`: the number of species

#     Returns:
#     - a named tuple with the parameters and the initial conditions (p, u0)
#     """
#     np = count_parameters(N)
#     p = Dict([Meta.parse(string("k_", i)) => p[i] for i in 1:np])
#     p = push!(p, Meta.parse("U") => 1.)
#     u0 = [Meta.parse(string("x_", i)) => 0. for i in 1:N]
#     return (p = p, u0 = u0)
# end

function perturbation_events(input, perturbation_list)
    return [[input + rand()*p - p/2] for p in perturbation_list]
end

# function dictator(C, p)
#     dict = Dict()
#     for i in 1:length(C.parameters)
#         dict[C.parameters[i]] = p[i]
#     end
#     return dict
# end

# using ModelingToolkit

# function reorder(C, par_v)
#     dict = Dict()
#     for i in 1:length(ModelingToolkit.parameters(C.ext_ode))
#         dict[ModelingToolkit.parameters(C.ext_ode)[i]] = par_v[i]
#     end
#     out = []
#     for i in 1:length(C.parameters)
#         push!(out, dict[C.parameters[i]])
#     end
#     return out
# end


# function dictator_u(C, u)
#     dict = Dict()
#     for i in 1:length(C.species)
#         dict[C.species[i]] = u[i]
#     end
#     for i in 1:length(C.control)
#         dict[C.control[i]] = u[i+length(C.species)]
#     end

#     for i in 1:size(C.sensitivity_variables)[1]
#         for j in 1:size(C.sensitivity_variables)[2]
#             dict[C.sensitivity_variables[i,j]] = u[(i-1)*size(C.sensitivity_variables)[1]+j+length(C.species)+length(C.control)]
#         end
#     end
#     println(dict)
#     return dict
# end

function make_base_problem(C, ode, state0, control0, parameters, t_shared, verbose=true, where0=nothing)
    """
    Assemble an ODEProblem from an ODE definition and the required parameters.

    Args:
    - `ode`: the ODE system
    - `state0`: the initial state
    - `control0`: the initial control
    - `parameters`: the parameters
    - `t_shared`: the shared time span
    - `verbose`: whether to save the trajectory at each step
    - `where0`: where to save the trajectory 

    Returns:
    - the ODEProblem
    """
    return ODEProblem(ode, vcat(state0[1:C.N], control0..., state0[C.N+1:end]), (0., t_shared), C.to_hidden_order(parameters), saveat=verbose ? true : where0, save_everystep=verbose)
end

function run_SF_CPU(C, base_problem, t_shared, t_unique, controls, reltol=1e-12, abstol=1e-12, verbose=true, where1=nothing)
    """
    More efficient implementation of the simulation when a step function is applied at a fized time as input perturbation

    Args:
    - `C`: the CRN object
    - `base_problem`: the base problem (in this way, we can avoid to repeat some unuseful computations)
    - `t_shared`: the time up to which the trajectory is unperturbed (shared by all the simulations)
    - `t_unique`: the time up to which the trajectory is perturbed (NOTE: t_unique>t_shared)
    - `controls`: the list of controls to apply (arrays of values)
    - `reltol`: the relative tolerance
    - `abstol`: the absolute tolerance
    - `verbose`: whether to save the trajectory at each step
    - `where1`: where to save the trajectory (for the perturbed part)

    Returns: (a pair)
    - the solution of the shared part of the trajectory
    - the solutions of the perturbed part of the trajectory
    """
    solutions = Array{Any}(undef, length(controls))
    shared_solution = solve(base_problem, Tsit5(), reltol=reltol, abstol=abstol)
    
    problem = remake(base_problem, tspan=(t_shared, t_unique), u0=shared_solution.u[end], saveat=verbose ? true : where1, save_everystep=verbose)

    Threads.@threads for i in 1:length(controls)
        # structure:                      STATE                        |   CONTROL     |             SENSITIVITIES
        problem = remake(problem, u0=vcat(shared_solution.u[end][1:C.N], controls[i]..., shared_solution.u[end][C.N+length([controls[i]...])+1:end]))
        sol = solve(problem, Tsit5(), reltol=reltol, abstol=abstol)
        solutions[i] = sol
    end
    return (shared_solution, solutions)
end


function runp_SF_CPU(C, base_problem, parameter_sets, t_shared, t_unique, controls, reltol=1e-12, abstol=1e-12, verbose=true, where1=nothing)
    """
    More efficient implementation of the simulation when a step function is applied at a fized time as input perturbation
    This version operates on parameter sets

    Args:
    - `C`: the CRN object
    - `base_problem`: the base problem (in this way, we can avoid to repeat some unuseful computations)
    - `parameter_sets`: operate on multiple parameter sets
    - `t_shared`: the time up to which the trajectory is unperturbed (shared by all the simulations)
    - `t_unique`: the time up to which the trajectory is perturbed (NOTE: t_unique>t_shared)
    - `controls`: the list of controls to apply (arrays of values)
    - `reltol`: the relative tolerance
    - `abstol`: the absolute tolerance
    - `verbose`: whether to save the trajectory at each step
    - `where1`: where to save the trajectory (for the perturbed part)

    Returns: (a pair)
    - the solution of the shared part of the trajectory
    - the solutions of the perturbed part of the trajectory
    """
    
    out = Array{Any}(undef, length(parameter_sets))
    for j in 1:length(parameter_sets)
        unperturbed_problem = remake(base_problem, p=parameter_sets[j])
        unperturbed_solution = solve(unperturbed_problem, Tsit5(), reltol=reltol, abstol=abstol)
        solutions = Array{Any}(undef, length(controls))
        
        problem = remake(unperturbed_problem, tspan=(t_shared, t_unique), u0=unperturbed_solution.u[end], saveat=verbose ? true : where1, save_everystep=verbose)

        Threads.@threads for i in 1:length(controls)
            # structure:                      STATE                        |   CONTROL     |             SENSITIVITIES
            problem = remake(problem, u0=vcat(unperturbed_solution.u[end][1:C.N], controls[i]..., unperturbed_solution.u[end][C.N+length([controls[i]...])+1:end]))
            sol = solve(problem, Tsit5(), reltol=reltol, abstol=abstol)
            solutions[i] = sol
        end
        out[j] = (unperturbed_solution, solutions)
    end
    return out
end

### Other utilities

function vec2mat(v)
    return mapreduce(permutedims, vcat, v)
end