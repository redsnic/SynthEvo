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
            m[line+1,column+1] = v[ (line)*C.number_of_parameters + (column) + C.N + 2 ]
            line += 1
            if line == N
                line = 0
                column += 1
            end
        end
    end
    return m
end


function jacobian_pars(ode_sys, loss_data, loss_derivatives, sol, target, t0, t1, pars_v, pars_l, f_ss, sensitivity_offset, original_variables)
    fwd_pass = Dict([
        # [
        #     o_t0 => sol(t0, idxs=original_variables[target]),
        #     o_t1 => sol(t1, idxs=original_variables[target]),
        #     o_t0pdt => sol(t0+sensitivity_offset, idxs=original_variables[target])
        # ]
        [at_t0[i] => sol(t0, idxs=original_variables[i]) for i in 1:length(at_t0)];
        [at_t1[i] => sol(t1, idxs=original_variables[i]) for i in 1:length(at_t1)];
        [at_t0_d[i] => sol(t0*f_ss, idxs=original_variables[i]) for i in 1:length(at_t0_d)];
        [at_t1_d[i] => sol(t0+(t1-t0)*f_ss, idxs=original_variables[i]) for i in 1:length(at_t1_d)];
        #other params 
        [
            dU => (sol(t0+1., idxs=U) - sol(t0-1., idxs=U)),
            p_s => sensitivity_offset
        ];
        [ w[i] => loss_data[i].weight for i in 1:length(loss_data)];
    ])
    evaluated_loss_derivatives = Dict([
        [at_t0[i] => Symbolics.substitute(loss_derivatives[at_t0[i]], fwd_pass) for i in 1:length(at_t0)];
        [at_t1[i] => Symbolics.substitute(loss_derivatives[at_t1[i]], fwd_pass) for i in 1:length(at_t1)];
        [at_t0_d[i] => Symbolics.substitute(loss_derivatives[at_t0_d[i]], fwd_pass) for i in 1:length(at_t0_d)];
        [at_t1_d[i] => Symbolics.substitute(loss_derivatives[at_t1_d[i]], fwd_pass) for i in 1:length(at_t1_d)]
    ])
    S_t0 = sensitivity_from_ode(ode_sys, sol, t0)
    S_t1 = sensitivity_from_ode(ode_sys, sol, t1)
    sensitivities =  # could have used get_S
        # [ i!=target ? 0. : evaluated_loss_derivatives[o_t0] for i in 1:length(at_t0)]'*S_t0 +
        # [ i!=target ? 0. : evaluated_loss_derivatives[o_t1] for i in 1:length(at_t0)]'*S_t1 +
        # [ i!=target ? 0. : evaluated_loss_derivatives[o_t0pdt] for i in 1:length(at_t0)]'*sensitivity_from_ode(ode_sys, sol, t0+sensitivity_offset) +
        [ evaluated_loss_derivatives[at_t0[i]] for i in 1:length(at_t0)]'*S_t0 +
        [ evaluated_loss_derivatives[at_t1[i]] for i in 1:length(at_t1)]'*S_t1 +
        [ evaluated_loss_derivatives[at_t0_d[i]] for i in 1:length(at_t0_d)]'*sensitivity_from_ode(ode_sys, sol, t0*f_ss) +
        [ evaluated_loss_derivatives[at_t1_d[i]] for i in 1:length(at_t1_d)]'*sensitivity_from_ode(ode_sys, sol, t0+(t1-t0)*f_ss)

    #L1_reg = loss_data[3].weight.*sign.(pars)/((1.0+sum(abs.(pars))).^2) # TODO hardcoded (and changed to 1/(1+|x|))
    L1_reg = loss_data[3].weight*sign.(pars_v) #./(1.0.+pars_v)./((1.0.+log.(1.0 .+ pars_v)).^2) # lograrithmic regularization 

    total_sensitivity = vec(sensitivities) + vec(L1_reg)

    # compute here the loss directly
    loss = 0
    loss_array = []
    for i in 1:length(loss_data)
        loss_el = Symbolics.substitute(loss_data[i].sym, merge(fwd_pass, unsym_dict(pars_l.p))).val*loss_data[i].weight
        loss += loss_el
        push!(loss_array, loss_el)
    end

    return (
        sensitivity = total_sensitivity,
        derivatives_of_loss = evaluated_loss_derivatives,
        regularization_term = L1_reg,
        fwd_pass = fwd_pass,
        loss = loss,
        loss_array = loss_array
    )
end

function prepare_args(sol, target, t0, t1, pars_l, weights, p, d, f_ss, norm_for_sensitivity_loss, norm_for_ss_loss, norm_for_adaptation_loss)
    return [
        (
            eval = sensitivity_loss_eval,
            sym = sensitivity_loss_symbolic(norm_for_sensitivity_loss),
            args = (sol, target, t0, d, p), # sol, target, t0, d, p
            weight = weights[1]
        ),
        (
            eval = steady_state_loss_eval,
            sym = steady_state_loss_symbolic(norm_for_ss_loss),
            args = (sol, t0, t1, f_ss), # sol, t0, t1, f_ss
            weight = weights[2]
        ),
        (
            eval = L1_loss_eval,
            sym = L1_loss_symbolic(N, pars_l.p), # N, pars
            args = ([pars_l.p]), # p
            weight = weights[3]
        ),
        (
            eval = adaptation_loss_eval,
            sym = adaptation_loss_symbolic(norm_for_adaptation_loss), # N
            args = (sol, target, t0, t1), # sol, target, t0, t1
            weight = weights[4]
        )
    ]
end


function update_args(sol, target, t0, t1, pars_l, old_args, p, d, f_ss)
    return [
        (
            eval = old_args[1].eval,
            sym = old_args[1].sym,
            args = (sol, target, t0, d, p), # sol, target, t0, d, p
            weight = old_args[1].weight
        ),
        (
            eval = old_args[2].eval,
            sym = old_args[2].sym,
            args = (sol, t0, t1, f_ss), # sol, t0, t1, f_ss
            weight = old_args[2].weight
        ),
        (
            eval = old_args[3].eval,
            sym = old_args[3].sym, # N, p
            args = ([pars_l.p]), # p
            weight = old_args[3].weight
        ),
        (
            eval = old_args[4].eval,
            sym = old_args[4].sym, # N
            args = (sol, target, t0, t1), # sol, target, t0, t1
            weight = old_args[4].weight
        )
    ]
end

function compute_symbolic_derivatives_of_loss(symbolic_loss)
    # the variables are assumed to be in the global environment
    loss_derivatives = Dict([
        # [
        #     o_t0 => (Symbolics.derivative(symbolic_loss, o_t0)),
        #     o_t1 => (Symbolics.derivative(symbolic_loss, o_t1)),
        #     o_t0pdt => (Symbolics.derivative(symbolic_loss, o_t0pdt))
        # ]
        [at_t0[i] => Symbolics.derivative(symbolic_loss, at_t0[i]) for i in 1:length(at_t0)];
        [at_t1[i] => Symbolics.derivative(symbolic_loss, at_t1[i]) for i in 1:length(at_t1)];
        [at_t0_d[i] => Symbolics.derivative(symbolic_loss, at_t0_d[i]) for i in 1:length(at_t0_d)];
        [at_t1_d[i] => Symbolics.derivative(symbolic_loss, at_t1_d[i]) for i in 1:length(at_t1_d)];
    ])
    return loss_derivatives
end



# ---


function symbolic_gradient_descent(p0, crn_info, gd_options, gd_perturbation_options, gd_loss_options)

    tspan = (gd_perturbation_options.t0, gd_perturbation_options.t1)
    pars_v = p0
    pars_l = assemble_opt_parameters_and_varables(pars_v, crn_info.N)
    loss_args = gd_perturbation_options.loss_blueprint

    # output data structures
    grad_history_ada = vec(zeros(crn_info.np))
    momentum_adam = vec(zeros(crn_info.np))
    velocity_adam = vec(zeros(crn_info.np))
    non_pruned_parameters = vec(zeros(crn_info.np)) # implement heuristic to prune parameters

    if gd_options.verbose
        loss_tape = []
        loss_tape_array = []
        parameter_tape = []
        gradient_tape = []
        optimizer_tape = []
        push!(parameter_tape, pars_v)
        push!(gradient_tape, zeros(crn_info.np))
        push!(optimizer_tape, zeros(crn_info.np))
    end

    # profiling @(100, 5) ~ 18s 
    for i in 1:gd_options.n_iter
        gradient = zeros(crn_info.np) # zero the gradient
        if gd_options.verbose
            loss_tape = push!(loss_tape, 0.)
            loss_tape_array = push!(loss_tape_array, zeros(gd_loss_options.n_losses))
        end
        if gd_options.use_random_perturbation
            for _ in 1:gd_perturbation_options.K
                # profiling @(100, 5) ~ 13.5s 
                sol = run_extended(crn_info.ext_ode, pars_v, pars_l, gd_perturbation_options.input, gd_perturbation_options.perturbation, gd_perturbation_options.t0, gd_perturbation_options.t1)      # change to be more flexible in the perturbation events
                # profiling @(100, 5) ~ 0s 
                loss_args = update_args(sol, crn_info.target, gd_perturbation_options.t0, gd_perturbation_options.t1, pars_l, loss_args, gd_loss_options.p, gd_loss_options.d, gd_loss_options.f_ss)   # prepare the input for the next step
                # profiling @(100, 5) ~ 3s
                # symbolically "backpropagate"
                # profiling @(100, 5) ~ 3s
                jacobian = jacobian_pars(crn_info.ext_ode, loss_args, gd_options.symbolic_derivatives_of_los, sol, crn_info.N, gd_perturbation_options.t0, gd_perturbation_options.t1, pars_v, pars_l, gd_loss_options.f_ss, gd_loss_options.d, [[Symbol("x_$(i)") for i in 1:N]..., Symbol("U")])
                # profiling @(100, 5) ~ 1s
                gradient += vec([v.val for v in jacobian.sensitivity])  # check efficiency with symbolic operations (maybe we have to use "real" types directly)
            
                if gd_options.verbose
                    loss_tape[end] += jacobian.loss                         # record the loss
                    loss_tape_array[end] = loss_tape_array[end] + jacobian.loss_array
                end
            end
        else
            solutions = run_extended_with_fixed_perturbations(crn_info.ext_ode, pars_l, gd_perturbation_options.input, gd_perturbation_options.perturbation_list, gd_perturbation_options.t0, gd_perturbation_options.t1)
            for sol in solutions
                loss_args = update_args(sol, crn_info.target, gd_perturbation_options.t0, gd_perturbation_options.t1, pars_l, loss_args, gd_loss_options.p, gd_loss_options.d, gd_loss_options.f_ss)   # prepare the input for the next step
                # symbolically "backpropagate"
                jacobian = jacobian_pars(crn_info.ext_ode, loss_args, gd_options.symbolic_derivatives_of_loss, sol, crn_info.N, gd_perturbation_options.t0, gd_perturbation_options.t1, pars_v, pars_l, gd_loss_options.f_ss, gd_loss_options.d, [[Symbol("x_$(i)") for i in 1:N]..., Symbol("U")])
                gradient += vec([v.val for v in jacobian.sensitivity])  # check efficiency with symbolic operations (maybe we have to use "real" types directly)
                if gd_options.verbose
                    loss_tape[end] += jacobian.loss                         # record the loss
                    loss_tape_array[end] = loss_tape_array[end] + jacobian.loss_array
                end
            end
        end
        if gd_options.use_random_perturbation
            if gd_options.verbose
                loss_tape[end] /= K # average the recorded loss
                loss_tape_array[end] ./= K
            end
            gradient /= K       # average the gradient
        else
            if gd_options.verbose
                loss_tape[end] /= length(gd_perturbation_options.perturbation_list) # average the recorded loss
                loss_tape_array[end] ./= length(gd_perturbation_options.perturbation_list)
            end
            gradient /= length(gd_perturbation_options.perturbation_list)       # average the gradient
        end

        if gd_options.use_pruning_heuristic 
            gradient = gradient .* non_pruned_parameters
        end

        if gd_options.clip_value != nothing
            gradient = max.(min.(gradient, clip_value), -clip_value)
        end
        if gd_options.use_gradient_normalization
            m = maximum(abs.(gradient))
            if m > 1.
                gradient /= m
            end
        end
        max_gradient = maximum(abs.(gradient))
        gradient = gradient + max_gradient/100.0.*randn(length(gradient)) .- max_gradient/200 # add some noise to the gradient
        if gd_options.use_adagrad
            lr = adagrad_update_get_coefficient(pars_v, gradient, grad_history_ada, gd_options.alpha)
            if gd_options.verbose
                push!(optimizer_tape, lr)
            end
        elseif gd_options.use_adam
            lr = ADAM_update_get_coefficient(pars_v, gradient, momentum_adam, velocity_adam, gd_options.alpha, 0.9, 0.9, 1e-8)
            if gd_options.verbose
                push!(optimizer_tape, lr)
            end
        else
            lr = gd_options.alpha
        end
        
        # update the parameters (avoid negative values)
        
        if gd_options.use_pruning_heuristic
            pars_v = max.(0., pars_v - (lr .* gradient)).*non_pruned_parameters
            non_pruned_parameters = (pars_v .> 0.0001)
        else
            pars_v = max.(0., pars_v - (lr .* gradient))
        end
        # update dictionary like parameters
        pars_l = assemble_opt_parameters_and_varables(pars_v, crn_info.N) 
        
        if gd_options.verbose
            push!(parameter_tape, pars_v)
            push!(gradient_tape, gradient)
        end
    end

    if !gd_options.verbose
        return (
            parameters = pars_v,
            loss_tape = nothing,
            loss_tape_array = nothing,
            parameter_tape = nothing,
            gradient_tape = nothing,
            optimizer_tape = nothing
        )
    else
        return (
            parameters = pars_v,
            loss_tape = loss_tape,
            loss_tape_array = loss_tape_array,
            parameter_tape = parameter_tape,
            gradient_tape = gradient_tape,
            optimizer_tape = optimizer_tape
        )
    end
end

using LinearAlgebra

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