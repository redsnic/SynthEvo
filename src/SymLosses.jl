

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
        #L = ifelse( dU < 1.0 , ((at_t0[3] - at_t1[3])/at_t0[3])^2, Num(0.) )  # not less then 0.5
        #L =  ifelse( dU < 1.0 , (max(0.,(at_t1[3] - 0.5)^3)), Num(0.) ) 
        #L2 = 10*abs(at_t0[3] - 0.5) # (0.25 - min(abs(at_t0[3]), 0.25))
        #return L/(1.0 + L) + L2/(1.0 + L2)
        #return L + L2 
        L =  ifelse(dU < 0.0,
            (at_t1[3]-0.25)^2,
            Num(0.) ) 
              
        #L2 = (at_t0[3]-0.25)^2
        return L #+ L2
    else
        # not implemented
        L = 0/0
        return L/(1.0 + L)
    end
    #return 1.0/((abs(o_t1-o_t0)/o_t1)/(dU)) 
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
        #L = ifelse(dU >= 1., -((at_t0[3] - at_t1[3])/at_t0[3])^2, Num(0.))
        L = ifelse(dU >= 0., 
            ((at_t1[3]-0.75)^2),
        Num(0.))
        return L # L/(1.0 + L) #((p_s*dU - min((abs(o_t0pdt - o_t0)), p_s*dU))/dU)*()  # abs(abs(o_t0pdt - o_t0) - p_s)  # abs(abs(o_t0pdt - o_t0) - p_s*dU)
    else
        # not implemented
        #L = (p_s - min((abs(o_t0pdt - o_t0)), p_s))^norm
        return L/(1.0 + L) # (abs(o_t0pdt - o_t0) - p_s*dU)^norm TODO: check this
    end
    #return abs(((o_t0pdt - o_t0)/o_t0)/(dU+1.))
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
        L = sum((at_t0 - at_t0_d).^2 + (at_t1 - at_t1_d).^2)
        return L# /(1.0 + L)
    else
        L = sum(abs.(at_t0 - at_t0_d).^norm) + sum(abs.(at_t1 - at_t1_d).^norm)
        return L /(1.0 + L)
    end
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
    p = [Symbolics.variable(k) for (k, v) in p if k != :U]
    #return sum(abs.(values(p))) / (1.0 + sum(abs.(values(p))))
    L = sum(p) #sum(log.(1.0 .+ p))
    return L#/(1.0 + L)
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