using SimpleDiffEq

N = 3
n_losses = 4
unsym = (x) -> eval(Meta.parse(string(x)))
unsym_dict = (d) -> Dict([unsym(k) => v for (k,v) in d])

function sensitivity(ode, parameters)
    S = Matrix{Num}(undef, length(equations(ode)), length(parameters)-1)
    J =  Matrix{Num}(undef, N, N)
    V = Matrix{Num}(undef, length(equations(ode)), length(parameters)-1)
    for (i, eq) in enumerate(equations(ode))
        # correction = 0
        # for (j, (k, v)) in enumerate(parameters)
        #     if isequal(k, :U)
        #         correction = 1
        #         continue
        #     else
        #         pos = parse(Int, split(string(k), "_")[2])
        #     end
        #     S[i, pos] = Symbolics.derivative(eq.rhs, eval(Meta.parse(string(k))))
        # end

        for j in 1:length(parameters)-1
            S[i, j] = Symbolics.derivative(eq.rhs,eval(Meta.parse(string("k_$j"))))
        end
        # jacobian 3x3
        for j in 1:N
            J[i, j] = Symbolics.derivative(eq.rhs,eval(Meta.parse(string("x_$j"))))
        end

        for j in 1:length(parameters)-1
            V[i, j] = eval(Meta.parse(string("ks_$(i)_$j")))
        end

    end
    return S + J*V
end
        
# adaptation_loss variables
o_t0 = Symbolics.variable(:o_t0)
o_t1 = Symbolics.variable(:o_t1)

# sensitivity loss variables
o_t0pdt = Symbolics.variable(:o_t0pdt)
o_t0 = Symbolics.variable(:o_t0)
dU = Symbolics.variable(:dU)
p_s = Symbolics.variable(:p_s) # proportionality factor

# steady_state_loss variables 
at_t0 = Symbolics.variables(:at_t0, 1:N)
at_t1 = Symbolics.variables(:at_t1, 1:N)
at_t0_d = Symbolics.variables(:at_t0_d, 1:N)
at_t1_d = Symbolics.variables(:at_t1_d, 1:N)

function adaptation_loss_symbolic(norm = 1)
    """
    Compute the absolute error of the adaptation. 
    This is the difference of the change in the target species at the two fix points.

    Args:
    - `L`: the norm to use

    Returns:
    - the absolute error
    """
    if norm == 1
        L = abs(o_t1 - o_t0) + 10*(0.5 - min(o_t0, 0.5))/(1.0 + (0.5 - min(o_t0, 0.5)))
        return L/(1.0 + L)
    else
        L = abs(o_t1 - o_t0)^norm #+ (0.5 - min(o_t0, 0.5))^norm
        return L/(1.0 + L)
    end
    #return 1.0/((abs(o_t1-o_t0)/o_t1)/(dU)) 
end

function adaptation_loss_eval(sym_expr, sol, target, t0, t1)
    """
    Compute the absolute error of the adaptation. 
    This is the difference of the change in the target species at the two fix points.

    Args:
    - `sol`: the solution of the ODE
    - `target`: the target species
    - `t0`: the first fix point (unperturbed)
    - `t1`: the second fix point (perturbed)
    - norm: the norm to use

    Returns:
    - the absolute error
    """
    return substitute(sym_expr, Dict(
        o_t0 => sol(t0, idxs=x_3),
        o_t1 => sol(t1, idxs=x_3),
        dU => abs(sol(t0-1., idxs=U) - sol(t0+1., idxs=U))
    ))
end

function sensitivity_loss_symbolic(norm = 1)
    """
    Compute the sensitivity loss of the CRN.
    This is the difference between the fixpoint of the unperturbed system and the 
    trainsient of perturbed system after a delta of time.

    Args:
    - `N`: the number of species

    Returns:
    - the sensitivity loss
    """
    if norm == 1
        L = (p_s - min((abs(o_t0pdt - o_t0)), p_s))
        return L/(1.0 + L) #((p_s*dU - min((abs(o_t0pdt - o_t0)), p_s*dU))/dU)*()  # abs(abs(o_t0pdt - o_t0) - p_s)  # abs(abs(o_t0pdt - o_t0) - p_s*dU)
    else
        L = (p_s - min((abs(o_t0pdt - o_t0)), p_s))^norm
        return L/(1.0 + L) # (abs(o_t0pdt - o_t0) - p_s*dU)^norm TODO: check this
    end
    #return abs(((o_t0pdt - o_t0)/o_t0)/(dU+1.))
end

function sensitivity_loss_eval(sym_expr, sol, target, t0, delta_t, proportion)
    """
    Compute the sensitivity loss of the CRN.
    This is the difference between the fixpoint of the unperturbed system and the 
    trainsient of perturbed system after a delta of time.

    Args:
    - `sol`: the solution of the ODE
    - `target`: the target species
    - `t0`: the fix point (unperturbed)
    - `delta_t`: the time after the perturbation
    - `proportion`: the proportionality factor

    Returns:
    - the sensitivity loss
    """
    return substitute(sym_expr, Dict(
        o_t0 => sol(t0)[target],
        o_t0pdt => sol(t0+delta_t)[target],
        o_t1 => sol(t1)[target],
        dU => abs(sol(t0-1., idxs=U) - sol(t0+1., idxs=U)),
        p_s => proportion
    ))
end

function steady_state_loss_symbolic(norm = 1)
    """
    Compute the steady state loss of the CRN.
    The system is supposed to reach fixpoints at t0 and t1.

    Args:
    - `sol`: the solution of the ODE
    - `N`: the number of species
    - `t0`: the first fix point
    - `t1`: the second fix point
    - `d`: the fraction of the time span to consider

    vars:
    - at_t0: the state at t0
    - at_t1: the state at t1
    - at_t0_d: the state at t0*d
    - at_t1_d: the state at t0+(t1-t0)*d

    Returns:
    - the steady state loss
    """
    if norm == 1
        L = sum(abs.(at_t0 - at_t0_d) + abs.(at_t1 - at_t1_d))/2
        return L/(1.0 + L)
    else
        L = sum(abs.(at_t0 - at_t0_d).^norm) + sum(abs.(at_t1 - at_t1_d).^norm)/2
        return L/(1.0 + L)
    end
end

function steady_state_loss_eval(sym_expr, sol, t0, t1, f_ss=0.5)
    """
    Compute the steady state loss of the CRN.
    The system is supposed to reach fixpoints at t0 and t1.

    Args:
    - `sol`: the solution of the ODE
    - `N`: the number of species
    - `t0`: the first fix point
    - `t1`: the second fix point
    - `d`: the fraction of the time span to consider

    Returns:
    - the steady state loss
    """
    vars = [x_1, x_2, x_3]
    return substitute(sym_expr, Dict([
        [at_t0[i] => sol(t0, idxs=vars[i]) for i in 1:length(at_t0)];
        [at_t1[i] => sol(t1, idxs=vars[i]) for i in 1:length(at_t1)];
        [at_t0_d[i] => sol(t0*f_ss, idxs=vars[i]) for i in 1:length(at_t0_d)];
        [at_t1_d[i] => sol(t0+(t1-t0)*f_ss, idxs=vars[i]) for i in 1:length(at_t1_d)]
    ]))
end

function L1_loss_symbolic(N, p)
    """
    Compute the L1 norm of the parameters.

    Args:
    - `p`: the parameters

    Returns:
    - the L1 norm
    """
    np = count_parameters(N)
    p = [Symbolics.variable(k) for (k, v) in p if k != :U && k != :k_29] # exclude x_3 degradation rate 
    #return sum(abs.(values(p))) / (1.0 + sum(abs.(values(p))))
    L = sum(log.(1.0 .+ p))
    return L/(1.0 + L)
end

function L1_loss_eval(sym_expr, p)
    """
    Compute the L1 norm of the parameters.

    Args:
    - `p`: the parameters

    Returns:
    - the L1 norm
    """
    return substitute(sym_expr, unsym_dict(p))
end

w = Symbolics.variables(:w, 1:n_losses)
function total_loss_symbolic(losses)
    """
    Compute the total loss of the CRN.

    Args:
    - `k` number of loss components

    Returns:
    - the total loss
    """
    w = Symbolics.variables(:w, 1:length(losses))
    l = [losses[i].sym for i in 1:length(losses)]
    return sum(w .* l)
end


function total_loss_eval(losses)
    """
    Compute the total loss of the CRN.

    Args:
    - `sol`: the solution of the ODE
    - `target`: the target species
    - `t0`: the first fix point (unperturbed)
    - `t1`: the second fix point (perturbed)
    - `p`: the parameters
    - `weights`: the weights for the different losses
    - `norm`: the norm to use

    Returns:
    - the total loss
    """
    l = [losses[i].eval(losses[i].sym, losses[i].args...) for i in 1:length(losses)]
    return ( 
        total = sum(l .* [losses[i].weight for i in 1:length(losses)]),
        array = l .* [losses[i].weight for i in 1:length(losses)]
    )   
end

Sval = (S, what) -> substitute(S, what)
to_state = (x) -> [Symbol(string("x_$i")) => x[i] for i in 1:length(x)-1]
get_S = (S, x) -> Sval(S, unsym_dict(to_state(x)))

function make_sensitivity_ode(ode_sys, pars)
    S = sensitivity(ode_sys, pars)
    @variables (ks(t))[1:size(S,1), 1:size(S,2)]
    @variables (x(t))[1:length(equations(ode_sys)), 1]
    eval(Meta.parse("@variables " * join([ifelse(i==size(S,1) && j==size(S,2), "", "ks_" * string(i) * "_" * string(j) *"(t), ") for i in 1:size(S,1) for j in 1:size(S,2)]) * "ks_" * string(size(S,1)) * "_" * string(size(S,2)) * "(t)"))
    eval(Meta.parse("@parameters " * join(["k_" * string(i) *", " for i in 1:size(S,2)]) * "k_" * string(N)))
    eval(Meta.parse("@variables " * join(["x_" * string(i) *"(t), " for i in 1:size(S,1)]) * "x_" * string(N) * "(t)"))
    @variables U(t)
    D = Differential(t)
    # convert to ODESystem
    eqs = []
    for i in 1:length(equations(ode_sys))
        push!(eqs, equations(ode_sys)[i])
    end
    push!(eqs, D(U) ~ Num(0))
    for i in 1:size(S,1)
        for j in 1:size(S,2)
            # if isequal(S[i,j], Num(0))
            #     continue
            # end
            push!(eqs, D(eval(V("ks", "$(i)_$(j)"))) ~ S[i,j])
        end
    end
    return @named senesitivity_ode = ODESystem(eqs, t)
end


function extend_u0(ode, u0, p, N)
    np = count_parameters(N)
    return (
        u0 = [ [  ode.var_to_name[k] => v for (k,v) in u0]..., [ ode.var_to_name[Symbol("ks_$(i)_$(j)")] => 0. for i in 1:N for j in 1:np ]...,  ode.var_to_name[:U] => 1.],
        p = [ ode.var_to_name[k] => v for (k,v) in p]  
    )   
end

# obsolete
function run(crn, pars_v, pars_l, input, perturation, t0, t1)
    p = pars_v
    pars = pars_l
    prob_p = pars
    max_t = t1
    condition = [t0]
    affect! = make_perturbation_event(input, perturbation)
    prob = ODEProblem(crn, prob_p.u0, (0., max_t), prob_p.p)
    ps_cb = PresetTimeCallback(condition, affect!)
    sol = solve(prob, Tsit5(), reltol=1e-15, abstol=1e-15, callback=ps_cb)
    return sol
end

#obsolete
function run_extended(ext_ode, pars_v, pars_l, input, perturbation, t0, t1)
    p = pars_v
    condition = [t0]
    named_u_p = extend_u0(ext_ode, pars_l.u0, pars_l.p, N)
    affect! = make_perturbation_event(input, perturbation)
    prob = ODEProblem(ext_ode, named_u_p.u0, (0., t1), named_u_p.p)
    ps_cb = PresetTimeCallback(condition, affect!)
    sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12, callback=ps_cb)
    #sol = label(sol, ext_ode.var_to_name)
    return sol
end

function run_with_fixed_perturbations(crn, pars_v, pars_l, input, perturbation_list, t0, t1)
    condition = [t0]
    solutions = Array{Any}(undef, length(perturbation_list))
    prob = ODEProblem(crn, pars_l.u0, (0., t1), pars_l.p)

    Threads.@threads for i in 1:length(perturbation_list)
        function affect!(integrator)
            integrator[:U] = max(0., input + perturbation_list[i])
        end
        ps_cb = PresetTimeCallback(condition, affect!)
        sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12, callback=ps_cb)
        solutions[i] = sol
    end
    return solutions
end

#CUDA version ?
# using CUDA, StaticArrays, DiffEqGPU
# function run_with_fixed_perturbations(crn, pars_v, pars_l, input, perturbation_list, t0, t1)
#     condition = [t0]
#     solutions = Array{Any}(undef, length(perturbation_list))
#     prob = ODEProblem(crn, pars_l.u0, (0., t1), pars_l.p)

#     Threads.@threads for i in 1:length(perturbation_list)
#         function affect!(integrator)
#             integrator[:U] = max(0., input + perturbation_list[i])
#         end
#         ps_cb = PresetTimeCallback(condition, affect!)
#         sol = solve(prob, GPUTsit5(), reltol=1e-12, abstol=1e-12, callback=ps_cb)
#         solutions[i] = sol
#     end
#     return solutions
# end

function run_extended_with_fixed_perturbations(ext_ode, pars_l, input, perturbation_list, t0, t1)

    condition = [t0]
    named_u_p = extend_u0(ext_ode, pars_l.u0, pars_l.p, N)
    prob = ODEProblem(ext_ode, named_u_p.u0, (0., t1), named_u_p.p)

    solutions = Array{Any}(undef, length(perturbation_list))

    Threads.@threads for i in 1:length(perturbation_list)
        perturbation = perturbation_list[i]
        function affect!(integrator)
            integrator[:U] = max(0., input + perturbation )
        end
        ps_cb = PresetTimeCallback(condition, affect!)
        sol = solve(prob, Tsit5(), reltol=1e-12, abstol=1e-12, callback=ps_cb)
        solutions[i] = sol
    end
    return solutions
end

#cuda version
# function run_extended_with_fixed_perturbations(ext_ode, pars_l, input, perturbation_list, t0, t1)

#     condition = [t0]
#     named_u_p = extend_u0(ext_ode, pars_l.u0, pars_l.p, N)
#     prob = ODEProblem(ext_ode, named_u_p.u0, (0., t1), named_u_p.p)

#     solutions = Array{Any}(undef, length(perturbation_list))

#     Threads.@threads for i in 1:length(perturbation_list)
#         perturbation = perturbation_list[i]
#         function affect!(integrator)
#             integrator[:U] = max(0., input + perturbation )
#         end
#         ps_cb = PresetTimeCallback(condition, affect!)
#         sol = solve(prob, GPUTsit5(), reltol=1e-12, abstol=1e-12, callback=ps_cb)
#         solutions[i] = sol
#     end
#     return solutions
# end


function state_from_ode(ode, sol, t)
    return sol(t, idxs=[Symbol("x_$(i)") for i in 1:N])
end

function control_from_ode(ode, sol, t)
    return sol(t, idxs=Symbol("U"))
end

function sensitivity_from_ode(ode, sol, t) # TODO linkerlinker FASTER VERSION!
    v = sol(t)
    # indexes = dict_indexes(ode)
    m = zeros(N, count_parameters(N))
    # k = N+2
    line = 0
    column = 0
    for i in 1:N
        for j in 1:count_parameters(N)
            m[line+1,column+1] = v[ (line)*np + (column) + N + 2 ]
            #println(line, " ", column, " ", i, " ", j, " ", m[i,j])
            line += 1
            if line == N
                line = 0
                column += 1
            end
        end
    end
    return m
end


function jacobian_pars(ode_sys, loss_data, loss_derivatives, sol, target, t0, t1, pars, f_ss, sensitivity_offset, original_variables)
    fwd_pass = Dict([
        [
            o_t0 => sol(t0, idxs=original_variables[target]),
            o_t1 => sol(t1, idxs=original_variables[target]),
            o_t0pdt => sol(t0+sensitivity_offset, idxs=original_variables[target])
        ]
        [at_t0[i] => sol(t0, idxs=original_variables[i]) for i in 1:length(at_t0)];
        [at_t1[i] => sol(t1, idxs=original_variables[i]) for i in 1:length(at_t1)];
        [at_t0_d[i] => sol(t0*f_ss, idxs=original_variables[i]) for i in 1:length(at_t0_d)];
        [at_t1_d[i] => sol(t0+(t1-t0)*f_ss, idxs=original_variables[i]) for i in 1:length(at_t1_d)];
        #other params 
        [
            dU => abs(sol(t0-0.5, idxs=U) - sol(t0+0.5, idxs=U)),
            p_s => sensitivity_offset
        ];
        [ w[i] => loss_data[i].weight for i in 1:length(loss_data)];
    ])
    evaluated_loss_derivatives = Dict([
        [
            o_t0 => Symbolics.substitute(loss_derivatives[o_t0], fwd_pass),
            o_t1 => Symbolics.substitute(loss_derivatives[o_t1], fwd_pass),
            o_t0pdt => Symbolics.substitute(loss_derivatives[o_t0pdt], fwd_pass)
        ]
        [at_t0[i] => Symbolics.substitute(loss_derivatives[at_t0[i]], fwd_pass) for i in 1:length(at_t0)];
        [at_t1[i] => Symbolics.substitute(loss_derivatives[at_t1[i]], fwd_pass) for i in 1:length(at_t1)];
        [at_t0_d[i] => Symbolics.substitute(loss_derivatives[at_t0_d[i]], fwd_pass) for i in 1:length(at_t0_d)];
        [at_t1_d[i] => Symbolics.substitute(loss_derivatives[at_t1_d[i]], fwd_pass) for i in 1:length(at_t1_d)]
    ])
    S_t0 = sensitivity_from_ode(ode_sys, sol, t0)
    S_t1 = sensitivity_from_ode(ode_sys, sol, t1)
    sensitivities =  # could have used get_S
        [ i!=target ? 0. : evaluated_loss_derivatives[o_t0] for i in 1:length(at_t0)]'*S_t0 +
        [ i!=target ? 0. : evaluated_loss_derivatives[o_t1] for i in 1:length(at_t0)]'*S_t1 +
        [ i!=target ? 0. : evaluated_loss_derivatives[o_t0pdt] for i in 1:length(at_t0)]'*sensitivity_from_ode(ode_sys, sol, t0+sensitivity_offset) +
        [ evaluated_loss_derivatives[at_t0[i]] for i in 1:length(at_t0)]'*S_t0 +
        [ evaluated_loss_derivatives[at_t1[i]] for i in 1:length(at_t1)]'*S_t1 +
        [ evaluated_loss_derivatives[at_t0_d[i]] for i in 1:length(at_t0_d)]'*sensitivity_from_ode(ode_sys, sol, t0*f_ss) +
        [ evaluated_loss_derivatives[at_t1_d[i]] for i in 1:length(at_t1_d)]'*sensitivity_from_ode(ode_sys, sol, t0+(t1-t0)*f_ss)

    #L1_reg = loss_data[3].weight.*sign.(pars)/((1.0+sum(abs.(pars))).^2) # TODO hardcoded (and changed to 1/(1+|x|))
    L1_reg = loss_data[3].weight./(1.0.+pars)./((1.0.+log.(1.0 .+ pars)).^2) # lograrithmic regularization 
    L1_reg[29] = 0. # exclude x_3 degradation rate

    total_sensitivity = vec(sensitivities) + vec(L1_reg)
    return (
        sensitivity = total_sensitivity,
        derivatives_of_loss = evaluated_loss_derivatives,
        regularization_term = L1_reg,
        fwd_pass = fwd_pass
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
        [
            o_t0 => (Symbolics.derivative(symbolic_loss, o_t0)),
            o_t1 => (Symbolics.derivative(symbolic_loss, o_t1)),
            o_t0pdt => (Symbolics.derivative(symbolic_loss, o_t0pdt))
        ]
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
                if gd_options.verbose
                    loss = total_loss_eval(loss_args)
                    loss_tape[end] += loss.total.val                        # record the loss
                    loss_tape_array[end] += loss.array
                end
                # symbolically "backpropagate"
                # profiling @(100, 5) ~ 3s
                jacobian = jacobian_pars(crn_info.ext_ode, loss_args, gd_options.symbolic_derivatives_of_los, sol, crn_info.N, gd_perturbation_options.t0, gd_perturbation_options.t1, pars_v, gd_loss_options.f_ss, gd_loss_options.d, [[Symbol("x_$(i)") for i in 1:N]..., Symbol("U")])
                # profiling @(100, 5) ~ 1s
                gradient += vec([v.val for v in jacobian.sensitivity])  # check efficiency with symbolic operations (maybe we have to use "real" types directly)
            end
        else
            solutions = run_extended_with_fixed_perturbations(crn_info.ext_ode, pars_l, gd_perturbation_options.input, gd_perturbation_options.perturbation_list, gd_perturbation_options.t0, gd_perturbation_options.t1)
            for sol in solutions
                loss_args = update_args(sol, crn_info.target, gd_perturbation_options.t0, gd_perturbation_options.t1, pars_l, loss_args, gd_loss_options.p, gd_loss_options.d, gd_loss_options.f_ss)   # prepare the input for the next step
                if gd_options.verbose
                    loss = total_loss_eval(loss_args)
                    loss_tape[end] = loss_tape[end] + loss.total.val              # record the loss
                    loss_tape_array[end] = loss_tape_array[end] + loss.array
                end
                # symbolically "backpropagate"
                jacobian = jacobian_pars(crn_info.ext_ode, loss_args, gd_options.symbolic_derivatives_of_loss, sol, crn_info.N, gd_perturbation_options.t0, gd_perturbation_options.t1, pars_v, gd_loss_options.f_ss, gd_loss_options.d, [[Symbol("x_$(i)") for i in 1:N]..., Symbol("U")])
                gradient += vec([v.val for v in jacobian.sensitivity])  # check efficiency with symbolic operations (maybe we have to use "real" types directly)
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