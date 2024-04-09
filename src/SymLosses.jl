function loss_wrapper(expression, parameters, times)
    """
    Add automatic differentiation to the loss function.

    Args:
    - `expression`: the expression to differentiate
    - `parameters`: the parameters of the expression (eg. the dynamics at a give time point)
    - `times`: the times at which the expression is evaluated

    Returns:
    - the loss function, a named tuple:
    - - `fun`: the function to evaluate the loss
    - - `fun_dx`: the function to evaluate the derivatives of the loss
    - - `expr`: the expression
    - - `parameters`: the parameters
    - - `sub`: the function to evaluate the loss and its derivatives using the efficient computational graph
    - - `times`: the times at which the dynamics need to be evaluated

    Note: using same names for the variables for the dynamics helps in optimizing the efficiency of the computation and 
    the visualization of the loss function's expression.

    TODO: now it is actually mandatory to use the same names for the variables for the dynamics accross different components of the loss.
    check the use of unique in weighted_loss
    """
    dx = der_(expression, parameters)

    fun = eval(build_function(expression, parameters))
    fun_dx = [eval(eval(build_function(dxi, parameters))[1]) for dxi in dx]

    loss = (fun=fun, fun_dx=fun_dx, expr=expression, parameters=parameters)

    sub = (parameters) -> sub_(loss, parameters)

    loss = (fun=fun, fun_dx=fun_dx, expr=expression, parameters=parameters, sub = sub, times=times)

    return loss
end

function adaptation_loss(C, norm, target, t0, t1)
    """
    Compute the error of the adaptation. 
    This is the difference of the change in the target species at the two fix points.

    > parameters: X_0, X_1 state at t0 and t1

    Args:
    - `norm`: the norm to use

    Returns:
    - expression/variables pair
    """
    @parameters X_0[1:C.N] X_1[1:C.N]
    if norm == 1
        expression = abs(X_1[target] - X_0[target])
    else
        expression = (X_1[target] - X_0[target])^norm
    end
    parameters = [X_0, X_1]
    times = [t0, t1]
    return loss_wrapper(expression, parameters, times)
end

function steady_state_loss(C, norm, t0, t1, t0mdt, t1mdt)
    """
    Compute the error of the adaptation. 
    This is the difference of the change in the target species at the two fix points.

    > parameters: X_0, X_1, X_0mdt, X_1mdt state at t0 and t1

    Args:
    - `C` : the chemical reaction network
    - `norm`: the norm to use
    - `t0`: the time at which the steady state value of the unperturbed system is evaluated (final)
    - `t1`: the time at which the steady state value of the perturbed system is evaluated (final)
    - `t0mdt`: the time at which the steady state value of the unperturbed system is evaluated (initial)
    - `t1mdt`: the time at which the steady state value of the perturbed system is evaluated (initial)

    Returns:
    - the loss tuple as defined by loss_wrapper
    """
    @parameters X_0[1:C.N] X_1[1:C.N] X_0mdt[1:C.N] X_1mdt[1:C.N]
    if norm == 1
        expression = abs(X_0[1] - X_0mdt[1]) + abs(X_0[2] - X_0mdt[2]) + abs(X_0[3] - X_0mdt[3]) + abs(X_1[1] - X_1mdt[1]) + abs(X_1[2] - X_1mdt[2]) + abs(X_1[3] - X_1mdt[3])
    else
        expression = (X_0[1] - X_0mdt[1])^norm + (X_0[2] - X_0mdt[2])^norm + (X_0[3] - X_0mdt[3])^norm + (X_1[1] - X_1mdt[1])^norm + (X_1[2] - X_1mdt[2])^norm + (X_1[3] - X_1mdt[3])^norm
    end
    parameters = [X_0, X_1, X_0mdt, X_1mdt]
    times = [t0, t1, t0mdt, t1mdt]
    return loss_wrapper(expression, parameters, times)
end

function sensitivity_loss(C, norm, target, min_response, t0, t0pdt)
    """
    Compute the sensitivity loss.

    > parameters: X_0, X_0pdt state at t0 and t0pdt

    Args:
    - `C` : the chemical reaction network
    - `norm`: the norm to use
    - `target`: the target species (index)
    - `min_response`: the minimum response seeked
    - `t0`: the time at which the steady state value of the unperturbed system is evaluated
    - `t0pdt`: the time at which the sensitivity is evaluated

    Returns:
    - the loss tuple as defined by loss_wrapper
    """
    @parameters X_0[1:C.N] X_0pdt[1:C.N] 
    if norm == 1
        expression = min_response - min(abs(X_0pdt[target] - X_0[target]), min_response)
    else
        expression = (min_response - min(abs(X_0pdt[target] - X_0[target]), min_response))^norm
    end
    parameters = [X_0, X_0pdt]
    times = [t0, t0pdt]
    return loss_wrapper(expression, parameters, times)
end

function regularization_loss(C, norm)
    """
    Compute the regularization loss.

    > parameters: parameters (reaction rates of the CRN)

    Args:
    - `C` : the chemical reaction network
    - `norm`: the norm to use

    Returns:
    - the loss tuple as defined by loss_wrapper

    NOTE: this loss does not pass through the dynamics and is directly applied to 
    the parameters of the CRN. This behavior can be enforced by using 'nothing' as the time.
    """
    if norm == 1
        expression = sum(abs.([C.parameters[i] for i in 1:length(C.parameters)]))
    else
        expression = sum([C.parameters[i] for i in 1:length(C.parameters)].^norm)
    end
    parameters = [C.parameters]
    times = [nothing]
    return loss_wrapper(expression, parameters, times)
end

function weighted_loss(losses, weights)
    """
    Combine multiple losses into a single loss.

    Args:
    - `losses`: a list of loss functions (wrapped in loss_wrapper)
    - `weights`: a list of weights

    Returns:
    - loss tuple as defined by loss_wrapper

    TODO: now it is actually mandatory to use the same names for the variables for the dynamics accross different components of the loss.
    check the use of unique in this function
    """
    expression = sum([losses[i].expr * weights[i] for i in 1:length(losses)])
    parameters = unique(vcat([losses[i].parameters for i in 1:length(losses)]...))
    times = unique(vcat([losses[i].times for i in 1:length(losses)]...))
    return loss_wrapper(expression, parameters, times)
end

function eval_loss(C, loss, CRN_parameters, trajectory, compute_derivatives=true)
    """
    Evaluate the loss function over a specific trajectory.

    Args:
    - `C`: the CRN object
    - `loss`: the loss function
    - `CRN_parameters`: the parameters of the CRN (used for e.g. regularization)
    - `trajectory`: the trajectory function
    - `compute_derivatives`: whether to compute the derivatives of the loss or not 

    """
    loss_inputs = []
    for t in loss.times
        if t == nothing
            push!(loss_inputs, CRN_parameters)
        else
            push!(loss_inputs, trajectory(t)[1:C.N+length(C.control)]) # TODO not situable for vectorized inputs as nested arrays (consider sum(length, x))
        end
    end
    loss_out = loss.sub(loss_inputs) # compute the value of the loss (aka forward pass)
    if !compute_derivatives # stop if we don't need the derivatives
        return (loss=loss_out.val, gradient=nothing)
    end

    # NOTE: it is assumed that the number of times is equal to the components of the loss
    # this does not mean we cannot support different equation at the same times, but that
    # would be however something that can be avoided while defining the losses.

    # compute the gradient of the loss wrt the CRN parameters
    sensitivities = zeros(length(CRN_parameters))
    for i in 1:length(loss.times)
        if loss.times[i] == nothing
            sensitivities += loss_out.der[i] # special case, direct derivative
        else
            loss_jac = loss_out.der[i]
            # chain rule to compute the gradient through the dynamics
            sensitivities += (vec(reshape(loss_jac, 1, length(loss_jac)) * sensitivity_from_ode(C, trajectory, loss.times[i]))) 
        end
    end
    return (loss=loss_out.val, gradient=sensitivities)
end

function der_(expression, parameters)
    """
    Compute the derivatives of the expression wrt the parameters.
    
    NOTE: might assume that the derivatives are wrt the CRN parameters directly or passing trough the ODE.

    Args:
    - `expression`: the expression to differentiate
    - `parameters`: the parameters of the expression

    Returns:
    - the (symbolic) derivatives as a nested array
    """
    dx = [[Symbolics.derivative(expression, x_i) for x_i in x] for x in parameters]
    return dx
end

function sub_(loss, parameters, compute_derivatives=true)
    """
    Evaluate the loss on specific parameters using the pre-computed computational graph.

    Args:
    - `loss`: the loss function (as from loss_wrapper)
    - `parameters`: the parameters at which to evaluate the loss
    - `compute_derivatives`: whether to compute the derivatives of the loss or not
        
    Returns:
    - the value of the loss and its derivatives (named tuple with fields val and der)
    """
    val = loss.fun(parameters)
    if !compute_derivatives
        return (val=val, der=nothing)
    end
    der = [f(parameters) for f in loss.fun_dx]
    return (val=val, der=der)
end


function sensitivity_from_ode(C, sol, t) 
    """
    Compute the sensitivity matrix at time t.

    Args:
    - `ode`: the ODE system
    - `sol`: the solution of the ODE

    Returns:
    - the sensitivity matrix
    """
    v = sol(t)
    return reshape(v[C.N+2:end], C.number_of_parameters, C.N)'
end

function merge_solutions(shared_solution, solution, idx, t0)
    """
    Helper function to merge two ODE solutions. 

    Args:
    - `shared_solution`: the shared solution
    - `solution`: the solution
    - `idx`: the index of the solution
    - `t0`: the time at which the solutions diverge

    Returns:
    - the merged solution (a function that returns the interpolated solution at a given time)

    TODO: for efficiency and accuracy, this function should be used to tell the integrator where to look for the solution
    """
    function access!(time)
        if time < t0
            return shared_solution(time)
        else
            return solution[idx](time)
        end
    end
    return access!
end