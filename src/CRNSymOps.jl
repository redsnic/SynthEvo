using SimpleDiffEq


function sensitivity(C::CRN)

    # retrieve the CRN information
    ode, parameters = C.ode, C.parameters
    sensitivity_variables = C.sensitivity_variables
    species = C.species
    N = C.N

    S = Matrix{Num}(undef, N, length(parameters)) # d(X)/d(theta)
    J = Matrix{Num}(undef, N, N) # jacobian
    V = Matrix{Num}(undef, N, length(parameters)) # d(X)/(d(theta)dt)

    eqs = equations(ode)
    for i in 1:N
        eq = eqs[i]
        # sensitivity 3xnp
        for j in 1:length(parameters)
            S[i, j] = Symbolics.derivative(eq.rhs, parameters[j])
        end
        # jacobian 3x3
        for j in 1:N
            J[i, j] = Symbolics.derivative(eq.rhs,species[j])
        end
        # sensitivity variables 3xnp
        for j in 1:length(parameters)
            V[i, j] = sensitivity_variables[i,j] # eval(Meta.parse(string("ks_$(i)_$j")))
        end

    end
    return S + J*V
end
        

function make_sensitivity_ode(C::CRN)
    """
    Extend the ODE system with the sensitivity equations.

    Args:
    - `C`: the CRN model

    Returns:
    - the sensitivity ODE system
    """

    S = sensitivity(C)

    D = Differential(C.time)
    # convert to ODESystem
    eqs = []
    for i in 1:length(equations(C.ode))
        push!(eqs, equations(C.ode)[i])
    end
    # push!(eqs, D(C.control) ~ Num(0))
    for i in 1:size(S,1)
        for j in 1:size(S,2)
            # if isequal(S[i,j], Num(0))
            #     continue
            # end
            push!(eqs, D(C.sensitivity_variables[i,j]) ~ S[i,j])
        end
    end
    return @named senesitivity_ode = ODESystem(eqs, C.time)
end

function add_control_equations(C::CRN, controls)
    """
    Extends the ODE system with the control equations.

    Args:
    - `C`: the CRN model
    - `controls`: the list of control variables

    Returns:
    - the ODE system with the control equations
    """

    eqs = []
    for i in 1:length(equations(C.ode))
        push!(eqs, equations(C.ode)[i])
    end

    D = Differential(C.time)

    for c in controls
        push!(eqs, D(c) ~ Num(0))
    end

    return @named control_ode = ODESystem(eqs, C.time)

end


function sensitivity_from_ode(C::CRN, sol, t) 
    """
    Compute the sensitivity matrix at time t.

    Args:
    - `ode`: the ODE system
    - `sol`: the solution of the ODE

    Returns: 
    - the sensitivity matrix
    """
    v = sol(t)
    # sensitivity matrix
    m = zeros(C.N, C.number_of_parameters)
    line = 0
    column = 0
    for i in 1:C.N
        for j in 1:C.number_of_parameters
            m[line+1,column+1] = v[ (line)*C.number_of_parameters + (column) + C.N + 1 + 1 ] # todo Multiple inputs
            line += 1
            if line == C.N
                line = 0
                column += 1
            end
        end
    end
    return m
end




function symbolic_gradient_descent(p0, C, gd_options, gd_perturbation_options, tape_checkpoint=nothing, dt=0.1)
    """

    Perform a symbolic gradient descent on the parameters of a CRN model.

    Args:
    - `p0`: the initial parameters
    - `C`: the CRN model
    - `gd_options`: the gradient descent options
    - - `n_iter`: the number of iterations
    - - `verbose`: whether to record the loss and the parameters at each iteration
    - - `use_random_perturbation`: whether to use random perturbations
    - - `use_pruning_heuristic`: whether to use a pruning heuristic
    - - `clip_value`: the value to clip the gradient (nothing for no clipping)
    - - `use_gradient_normalization`: whether to normalize the gradient
    - - `use_gradient_noise`: whether to add noise to the gradient
    - - `alpha`: the base learning rate
    - - `use_adam`: whether to use adam 
    - - `use_adagrad`: whether to use adagrad (overrides use_adam, standard GD is the default)
    - `gd_perturbation_options`: the perturbation options
    - - `t0`: final time of the unperturbed trajectory
    - - `t1`: final time of the perturbed trajectory
    - - `loss_fun`: the loss function (wrapped in loss_wrapper)
    - - `input`: the input (initial control)
    - - `perturbation_list`: the list of perturbations (if random pertubations sample U(0,perturbation_list[i]) around the input)
    - `tape_checkpoint` (optional): continue writing on previosly defined output tapes
    - `dt`: the time step for the ODE solver (default 0.1)

    # TODO you might also wanto to save the state of the optimizer for checkpointing. (not needed)

    Returns:
    - a named tuple:
    - - `parameters`: the optimized parameters
    # if verbose is false, the following fields are `nothing`
    - - `loss_tape`: the loss at each iteration
    - - `parameter_tape`: the parameters at each iteration
    - - `gradient_tape`: the gradient at each iteration
    - - `optimizer_tape`: the learning rate at each iteration
    """

    tspan = (gd_perturbation_options.t0, gd_perturbation_options.t1)
    loss_fun = gd_perturbation_options.loss_fun 

    # output data structures
    grad_history_ada = vec(zeros(C.number_of_parameters))
    momentum_adam = vec(zeros(C.number_of_parameters))
    velocity_adam = vec(zeros(C.number_of_parameters))
    non_pruned_parameters = vec(zeros(C.number_of_parameters)) # implement heuristic to strongly prune parameters

    # setup opt problem
    base_problem = SynthEvo.make_base_problem(C, C.ext_ode, [0. for _ in 1:(length(C.species) + length(C.sensitivity_variables))], [gd_perturbation_options.input], p0, gd_perturbation_options.t0, true, dt)

    if gd_options.verbose
        if tape_checkpoint == nothing
            loss_tape = []
            # loss_tape_array = [] # TODO: currently unsupported
            parameter_tape = []
            gradient_tape = []
            optimizer_tape = []
            push!(parameter_tape, p0)
            push!(gradient_tape, zeros(C.number_of_parameters))
            push!(optimizer_tape, zeros(C.number_of_parameters))
        else
            loss_tape = tape_checkpoint.loss_tape
            # loss_tape_array = tape_checkpoint.loss_tape_array
            parameter_tape = tape_checkpoint.parameter_tape
            gradient_tape = tape_checkpoint.gradient_tape
            optimizer_tape = tape_checkpoint.optimizer_tape
        end
    end

    for i in 1:gd_options.n_iter
        gradient = zeros(C.number_of_parameters) # zero the gradient
        if gd_options.verbose
            loss_tape = push!(loss_tape, 0.)
            # loss_tape_array = push!(loss_tape_array, zeros(gd_loss_options.n_losses))
        end

        # prepare perturbations
        perturbation_list = nothing
        if gd_options.use_random_perturbation
            # here .perturbation_list contains the mean values of the perturbations
            perturbation_list = perturbation_events(gd_perturbation_options.input, gd_perturbation_options.perturbation_list)
        else
            perturbation_list = gd_perturbation_options.perturbation_list
        end
            
        # compute trajectories
        problem = remake(base_problem, p = C.to_hidden_order(p0))
        # TODO hardcoded parameters
        shared_solution, solutions = run_SF_CPU(C, problem, gd_perturbation_options.t0, gd_perturbation_options.t1, perturbation_list, 1e-5, 1e-5, true, 0.01) 
        
        # compute loss and gradient
        outs = []
        for i in 1:length(solutions)
            sol = merge_solutions(shared_solution, solutions, i, gd_perturbation_options.t0)
            loss = eval_loss(C, loss_fun, p0, sol, true) # .loss, .gradient
            push!(outs, loss)
            if gd_options.verbose
                loss_tape[end] += loss.loss
            end
        end
        gradient = sum([out.gradient for out in outs])
    
        # compute average gradient
        if gd_options.verbose
            loss_tape[end] /= length(gd_perturbation_options.perturbation_list) # average the recorded loss
            # loss_tape_array[end] ./= length(gd_perturbation_options.perturbation_list)
        end
        gradient /= length(gd_perturbation_options.perturbation_list)       # average the gradient
        
        # pruning
        if gd_options.use_pruning_heuristic 
            gradient = gradient .* non_pruned_parameters
        end

        # clipping
        if gd_options.clip_value != nothing
            gradient = max.(min.(gradient, clip_value), -clip_value)
        end

        # normalization
        if gd_options.use_gradient_normalization
            m = maximum(abs.(gradient))
            if m > 1.
                gradient /= m
            end
        end

        # noise
        if gd_options.use_gradient_noise
            # add some noise to the gradient
            max_gradient = maximum(abs.(gradient))
            gradient = gradient + max_gradient*gd_options.fraction_gradient_noise.*randn(length(gradient)) .- max_gradient/2*gd_options.fraction_gradient_noise 
        end

        # compute the learning rate
        # ADAGRAD
        if gd_options.use_adagrad
            lr = adagrad_update_get_coefficient(p0, gradient, grad_history_ada, gd_options.alpha)
            if gd_options.verbose
                push!(optimizer_tape, lr)
            end
        # ADAM
        elseif gd_options.use_adam
            lr = ADAM_update_get_coefficient(p0, gradient, momentum_adam, velocity_adam, gd_options.alpha, gd_options.ADAM_beta1, gd_options.ADAM_beta2, 1e-8)
            if gd_options.verbose
                push!(optimizer_tape, lr)
            end
        # pure GD
        else
            lr = gd_options.alpha
        end
        
        # update the parameters (avoid negative values)
        # pruning heuristic
        if gd_options.use_pruning_heuristic
            p0 = max.(0., p0 - (lr .* gradient)).*non_pruned_parameters
            non_pruned_parameters = (p0 .> 0.0001)
        else
            p0 = max.(0., p0 - (lr .* gradient))
        end
        
        # store results
        if gd_options.verbose
            push!(parameter_tape, p0)
            push!(gradient_tape, gradient)
        end
    end

    if !gd_options.verbose
        return (
            parameters = p0,
            loss_tape = nothing,
            # loss_tape_array = nothing,
            parameter_tape = nothing,
            gradient_tape = nothing,
            optimizer_tape = nothing
        )
    else
        return (
            parameters = p0,
            loss_tape = loss_tape,
            # loss_tape_array = loss_tape_array,
            parameter_tape = parameter_tape,
            gradient_tape = gradient_tape,
            optimizer_tape = optimizer_tape
        )
    end
end

# ---- TODO

function joint_jacobian(i, j, jac, initial_conditions)
    A_ij = substitute(jac[i, j], unsym_dict(initial_conditions))
    return A_ij
end

function compute_homeostatic_coefficient(crn, jac, pars, t0 = 10., t1=20, input = 1, perturb = 1.)
        
    opt_pars_v = pars
    opt_pars_l = assemble_opt_parameters_and_varables(opt_pars_v, N)

    jac = Symbolics.substitute(jac, unsym_dict(opt_pars_l.p))
                                                                 
    steady_state_after_perturbation = run_with_fixed_perturbations(crn, opt_pars_v, opt_pars_l, input, [ perturb ], t0, t1)[1](t1)[1:3]
    steady_state_after_perturbation = [
        :x_1 => steady_state_after_perturbation[1],
        :x_2 => steady_state_after_perturbation[2],
        :x_3 => steady_state_after_perturbation[3]
    ]

    A_21 = joint_jacobian(2, 1, jac, steady_state_after_perturbation)
    A_32 = joint_jacobian(3, 2, jac, steady_state_after_perturbation)
    A_22 = joint_jacobian(2, 2, jac, steady_state_after_perturbation)
    A_31 = joint_jacobian(3, 1, jac, steady_state_after_perturbation)

    # println("A_21 = ", A_21)
    # println("A_32 = ", A_32)
    # println("A_22 = ", A_22)
    # println("A_31 = ", A_31)
    # println("A_22*A_31 = ", A_22*A_31)
    # println("A_21*A_32 = ", A_21*A_32)
    # println("A_22*A_31 - A_21*A_32 = ", A_22*A_31 - A_21*A_32)

    return (
        coefficient = A_22*A_31 - A_21*A_32,
        A_21 = A_21,
        A_32 = A_32,
        A_22 = A_22,
        A_31 = A_31,
        A_22_A_31 = A_22*A_31,
        A_21_A_32 = A_21*A_32
    )
end

# function compute_homeostatic_coefficient(par_list)
#     out = []
#     for pars in par_list
#         !push(out, compute_homeostatic_coefficient(pars))
#     end
#     return out
# end