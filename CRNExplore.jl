using DifferentialEquations, RecursiveArrayTools, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO, OptimizationMOI, Ipopt
using CSV, DataFrames, Plots
using Catalyst, Combinatorics, GraphViz, Symbolics


function count_parameters(N)
    """
    Count the number of parameters in the CRN model

    Args:
    - N: number of species (the emptyset is not counted)

    Returns:
    - the number of parameters in the CRN model
    """
    res = 0
    # 0 -> 1
    res += N-1 # no x_1 as it is managed by the input
    # 0 -> 2
    res += length(combinations(1:N, 2)) # remove nothing -> 2A
    # 1 -> 1
    res += N*(N-1)
    # 1 -> 2
    res += N*length(with_replacement_combinations(1:N, 2)) - N # remove A -> 2A
    # 1 -> 0
    res += N
    # 2 -> 0
    res += length(with_replacement_combinations(1:N, 2))
    # 2 -> 1
    res += length(with_replacement_combinations(1:N, 2))*N - N # remove 2A -> A
    # 2 -> 2
    res += length(with_replacement_combinations(1:N, 2))*length(with_replacement_combinations(1:N, 2)) - length(with_replacement_combinations(1:N, 2)) # remove X+Y -> X+Y
    return res
end

function V(basename, zzz)
    """
    Function to create a variable given a basename and a number/suffix

    Args:
    - basename: the base name of the variable
    - zzz: the suffix of the variable

    Returns:
    - the variable
    """
    return Meta.parse("$(basename)_$(zzz)")
end

@variables t # try to understand why it needs to be global...
@variables U(t)
function create_reactions(N)
    """
    Use Catalyst inteface to create the reactions of the fully-connected CRN model 
    with N species

    Args:
    - N: number of species

    Returns:
    - reactions: the list of reactions
    """
    np = count_parameters(N)
    #@parameters u, K[1:count_parameters(N)]
    eval(Meta.parse("@parameters " * join(["k_" * string(i) *", " for i in 1:np]) * "k_" * string(N)))
    #@species x_1(t), x_2(t), x_3(t) ...
    eval(Meta.parse("@species " * join(["x_" * string(i) *"(t), " for i in 1:(N-1)]) * "x_" * string(N) * "(t)"))


    reactions = []
    global_counter = 1

    push!(reactions, Reaction(U, nothing, [x_1]))
    #0 -> 1
    for i in 2:N
        push!(reactions, Reaction(eval(V("k",global_counter)), nothing, [eval(V("x",i))]))
        global_counter += 1
    end
    # 0 -> 2
    for (i,j) in combinations(1:N, 2)
        if i != j
            push!(reactions, Reaction(eval(V("k",global_counter)), nothing, [eval(V("x",i)),eval(V("x",j))]))
        else
            push!(reactions, Reaction(eval(V("k",global_counter)), nothing, [eval(V("x",i))], nothing, [2]))
        end
        global_counter += 1
    end
    # # # 1 -> 1
    for i in 1:N
        for j in 1:N
            if i != j
                push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i))], [eval(V("x",j))]) )
                global_counter += 1
            end
        end
    end
    # # 1 -> 2
    for i in 1:N
        for (j,k) in with_replacement_combinations(1:N, 2)
            if !(i == j == k)
                if j != k
                    push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i))], [eval(V("x",j)), eval(V("x",k))]))
                else
                    push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i))], [eval(V("x",j))], [1], [2]))
                end
                global_counter += 1
            end
        end
    end
    # # # 1 -> 0
    for i in 1:N
        push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i))], nothing))
        global_counter += 1
    end
    # # # 2 -> 0
    for (i,j) in with_replacement_combinations(1:N, 2)
        if i != j
            push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i)), eval(V("x",j))], nothing))
        else
            push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i))], nothing, [2], nothing))
        end
        global_counter += 1
    end
    # # # 2 -> 1
    for (i,j) in with_replacement_combinations(1:N, 2)
        for k in 1:N
            if !(i == j == k)
                if i != j
                    push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i)), eval(V("x",j))], [eval(V("x",k))]))
                else
                    push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i))], [eval(V("x",k))], [2], [1]))
                end
                global_counter += 1
            end
        end
    end
    # # # 2 -> 2
    for (i,j) in with_replacement_combinations(1:N, 2)
        for (k,l) in with_replacement_combinations(1:N, 2)
            if !(i == k && j == l)
                if i != j && k != l 
                    push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i)), eval(V("x",j))], [eval(V("x",k)), eval(V("x",l))]))
                elseif i == j && k != l
                    push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i))], [eval(V("x",k)), eval(V("x",l))], [2], [1,1]))
                elseif i != j && k == l
                    push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i)), eval(V("x",j))], [eval(V("x",k))], [1,1], [2]))
                else
                    push!(reactions, Reaction(eval(V("k",global_counter)), [eval(V("x",i))], [eval(V("x",k))], [2], [2]))
                end
                global_counter += 1
            end
        end
    end
    @named crn = ReactionSystem(reactions, t)

    return crn
end


function make_perturbation_event(input, intensity)
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

function integrate(prob, tspan, callback, reltol=1e-8, abstol=1e-8, maxiters=100)
    """
    Integrate the problem with a callback function 
    that will be called only halfway through the time span.

    Args:
    - `prob`: the ODE problem
    - `tspan`: the time span
    - `callback`: the callback function

    Returns:
    - the result of the numerical integration
    """
    condition = [(tspan[2] - tspan[1])/2]
    ps_cb = PresetTimeCallback(condition, callback)
    sol = solve(prob, Tsit5(), reltol=reltol, abstol=abstol, callback=ps_cb, maxiters=maxiters)
end

function adaptation_loss_RE(sol, target, t0, t1) 
    """
    Compute the relative error of the adaptation. 
    This is the fraction of the change in the target species at the two fix points.

    Args:
    - `sol`: the solution of the ODE
    - `target`: the target species
    - `t0`: the first fix point (unperturbed)
    - `t1`: the second fix point (perturbed)

    Returns:
    - the relative error
    """
    return abs(sol(t1)[target] - sol(t0)[target])/((sol(t0)[target]+sol(t1)[target]+0.001)/2)
end

function adaptation_loss(sol, target, t0, t1, norm = 1) 
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
    if norm == 1
        return abs(sol(t1)[target] - sol(t0)[target])
    else
        return ((abs(sol(t1)[target] - sol(t0)[target]))^L)^(1/L)
    end
end

function sensitivity_loss(sol, target, t0, delta_t = 1., offset = 0.2)
    """
    Compute the sensitivity loss of the CRN.
    This is the difference between the fixpoint of the unperturbed system and the 
    trainsient of perturbed system after a delta of time.

    Args:
    - `sol`: the solution of the ODE
    - `target`: the target species
    - `t0`: the fix point (unperturbed)
    - `delta_t`: the time after the perturbation

    Returns:
    - the sensitivity loss
    """
    return abs(abs(sol(t0+delta_t)[target] - sol(t0)[target]) - offset*(abs(sol(t0+delta_t)[end] - offset*sol(t0-delta_t)[end])))
end

function steady_state_loss(sol, N, t0, t1, d=0.5)
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
    return sum(abs.(sol(t0) - sol(t0*d)) + abs.(sol(t1) - sol(t0+(t1-t0)*d)))
end

function L1_loss(p)
    """
    Compute the L1 norm of the parameters.

    Args:
    - `p`: the parameters

    Returns:
    - the L1 norm
    """
    return sum(abs.(values(p)))
end

function full_loss(params, sol, target, t_bar, t1, sensitivity_dt, sensitivity_offset, steady_d, l1=10.0, l2=0.01, l3=1.0, l4=1., absolute=false)
    """
    Compute the full loss of the CRN.

    Args:
    - `params`: the parameters
    - `sol`: the solution of the ODE
    - `target`: the target species
    - `t_bar`: the fix point
    - `t1`: the perturbed fix point
    - `sensitivity_dt`: the time after the perturbation for the sensitivity loss
    - `sensitivity_offset`: the offset for the sensitivity loss 
    - `steady_d`: the fraction of the time span to consider for the steady state loss
    - `l1`: the weight of the adaptation loss
    - `l2`: the weight of the sensitivity loss
    - `l3`: the weight of the L1 norm of the parameters
    - `l4`: the weight of the steady state loss
    - `absolute`: whether to use the absolute or relative adaptation loss

    Returns:
    - the full loss
    """
    if absolute
        al = adaptation_loss
    else
        al = adaptation_loss_RE
    end
    return l1*al(sol, target, t_bar, t1) + l2*sensitivity_loss(sol, target, t_bar, sensitivity_dt, sensitivity_offset) + l3*L1_loss(params) + l4*steady_state_loss(sol, target, t_bar, t1, steady_d)
end

function stochastic_loss(K, prob, pertub, tspan, target, t_bar, t1, sensitivity_dt, sensitivity_offset, steady_d, l1=10.0, l2=0.01, l3=1.0, l4=1., absolute=false, reltol=1e-8, abstol=1e-8, maxiters=100)
    """
    Compute the stochastic loss of the CRN.

    Args:
    - `K`: the number of iterations (samples)
    - `prob`: the problem
    - `pertub`: the perturbation function
    - `tspan`: the time span
    - `target`: the target species
    - `t_bar`: the fix point
    - `t1`: the perturbed fix point
    - `sensitivity_dt`: the time after the perturbation for the sensitivity loss
    - `sensitivity_offset`: the offset for the sensitivity loss
    - `steady_d`: the fraction of the time span to consider for the steady state loss
    - `l1`: the weight of the adaptation loss
    - `l2`: the weight of the sensitivity loss
    - `l3`: the weight of the L1 norm of the parameters
    - `l4`: the weight of the steady state loss

    Returns:
    - the stochastic loss function
    """
    function p_loss(p)
        #  K=K, prob=prob, perturb!=perturb!, tspan=tspan, target=target, t_bar=t_bar, t1=t1, sensitivity_dt=sensitivity_dt, sensitivity_offset=sensitivity_offset, steady_d=steady_d, l1=l1, l2=l2, l3=l3, l4=l4
        prob = remake(prob; p = p)
        loss = 0.
        Threads.@threads for i in 1:K # Threads.@threads # the problem is actually the Vector not the threading... cool. TODO: fix that
            sol = integrate(prob, tspan, perturb!, reltol, abstol, maxiters)
            loss += full_loss(p, sol, target, t_bar, t1, sensitivity_dt, sensitivity_offset, steady_d, l1, l2, l3, l4, absolute)
        end
        return sum(loss)/K
    end

    return p_loss
end

function gradient_descent(ALPHA, NITER, pinit, stoch_loss_f, clip=true, extra_info=false, grad_clip=nothing, use_adagrad=true, kill_parameter_on_clip=true)
    """
    Perform a gradient decent on the loss function.

    Args:
    - `ALPHA`: the learning rate
    - `NITER`: the number of iterations
    - `pinit`: the initial parameters
    - `stoch_loss_f`: the stochastic loss function
    - `clip`: whether to clip the parameters to positive values
    - `extra_info`: whether to return extra information (losses, gradients, parameters)

    Returns:
    - a named tuple with the minimizer, the losses, the gradients and the parameters (minimizer, losses, grads, pars)
    """
    if extra_info 
        losses = zeros(NITER)
        grads = []
        ps = []
    end
    if use_adagrad
        grad_history = zeros(length(pinit))
    end
    if kill_parameter_on_clip
        valid_parameters = ones(length(pinit))
    end
    for it in 1:NITER
        grad = ForwardDiff.gradient(stoch_loss_f, pinit)
        if kill_parameter_on_clip
            grad = grad.*valid_parameters
        end
        if use_adagrad
            ada_ALPHA = adagrad_update_get_coefficient(pinit, grad, grad_history, ALPHA)
        end
        if grad_clip != nothing
            grad = sign.(grad).*min.(abs.(grad), grad_clip)
        end
        if use_adagrad
            pinit -= ada_ALPHA.*grad
        else
            pinit -= ALPHA.*grad
        end
        if clip
            pinit = max.(0., pinit) # clip to positive values
            if kill_parameter_on_clip
                valid_parameters = min.(valid_parameters, valid_parameters.*sign.(pinit)) # if 0. stay 0.
            end
        end
        if extra_info
            losses[it] = stoch_loss_f(pinit)
            push!(grads, grad)
            push!(ps, pinit)
        end
    end
    if extra_info
        return (minimizer = pinit, losses = losses, grads = grads, pars = ps)
    end
    return (minimizer = pinit, losses = nothing, grads = nothing, pars = nothing)
end

function assemble_opt_parameters_and_varables(p, N)
    """
    Assemble the optimization parameters and variables.
    It will return a 0. initial condition for all the species.

    Args:
    - `p`: the parameters
    - `N`: the number of species

    Returns:
    - a named tuple with the parameters and the initial conditions (p, u0)
    """
    np = count_parameters(N)
    p = Dict([Meta.parse(string("k_", i)) => p[i] for i in 1:np])
    p = push!(p, Meta.parse("U") => 1.)
    u0 = [Meta.parse(string("x_", i)) => 0. for i in 1:N]
    return (p = p, u0 = u0)
end

function adagrad_update_get_coefficient(p, g, h, lr, eps=1e-8)
    """
    Perform an adagrad update on the parameters.

    Args:
    - `p`: the parameters
    - `g`: the gradients
    - `h`: the history of the gradients
    - `eps`: the epsilon value

    Returns:
    - the updated lr (also modifies h in place)
    """
    h[:] = g.^2 + h
    return (lr)./(sqrt.(h) .+ eps)
end

function ADAM_update_get_coefficient(p, g, m, v, lr, iter, beta1=0.9, beta2=0.9, eps=1e-3)
    m[:] = beta1.*m + ((1-beta1).*g)
    v[:] = beta2.*v + ((1-beta2).*(g.^2))
    m_hat = m./(1-beta1^(iter))
    v_hat = v./(1-beta2^(iter))
    return lr.*(m_hat./sqrt.(v_hat) .+ eps)
end


### Other utilities

function vec2mat(v)
    return mapreduce(permutedims, vcat, v)
end