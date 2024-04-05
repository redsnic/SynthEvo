function adaptation_loss(norm, target)
    """
    Compute the error of the adaptation. 
    This is the difference of the change in the target species at the two fix points.

    > parameters: X_0, X_1 state at t0 and t1

    Args:
    - `norm`: the norm to use

    Returns:
    - expression/variables pair
    """
    @parameters X_0 X_1
    if norm == 1
        expression = abs(X_0[target] - X_1[target])
    else
        expression = (X_1[target] - X_0[target])^norm
    end
    J = Symbolics.jacobian(expression, [X_0 X_1])
    return (expr=expression, parameters=[X_0, X_1], J = J)
end


function sensitivity_loss(norm, min_sensitivity, target)
    """
    Soft constraint for the sensitivity.

    > parameters: X_0, X_d0 state at t0 and t0+dt

    Args:
    - `norm`: the norm to use

    Returns:
    - expression/variables pair
    """
    @parameters X_0 X_d0
    if norm == 1
        expression = abs(abs(X_0[target] - X_d0[target]) - min_sensitivity)
    else
        expression = (abs(X_0[target] - X_d0[target]) - min_sensitivity)^norm
    end
    return (expr=expression, variables=[X_0 X_d0])
end

function steady_state_loss(norm)
    """
    Compute the steady state loss of the CRN.
    The system is supposed to reach fixpoints at t0 and t1.

    Args:
    - `norm`: the norm to use

    Returns:
    - expression/variables pair
    """
    @parameters X_0 X_1 X_fp0 X_fp1
    if norm == 1
        expression = abs(X_0 - X_fp0) + abs(X_1 - X_fp1)
    else
        expression = (X_0 - X_fp0)^norm + (X_1 - X_fp1)^norm
    end
    return (expr=expression, parameters=[X_0, X_1, X_fp0, X_fp1])
end

function L1()
    """
    Compute the L1 norm of the parameters.

    Args:
    - `p`: the parameters

    Returns:
    - the L1 norm
    """
    @parameters p
    return (expr = sum(p), parameters = [p])
end

function weighted_loss(losses)
    """
    Compute the total loss of the CRN.

    Args:
    - `losses`: the loss components
    - `weights`: the weights of the loss components

    Returns:
    - the total loss
    """
    @parameters w[1:length(losses)]
    return (expr = sum(w .* losses), parameters = [w])
end


function total_loss_symbolic(losses) # weighted loss
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