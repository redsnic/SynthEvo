# Description: This file contains utility function for optimizing CRN models.

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

function ADAM_update_get_coefficient(p, g, m, v, lr, iter, beta1=0.9, beta2=0.9, eps=1e-6)
    """
    Perform an ADAM update on the parameters.

    Args:
    - `p`: the parameters
    - `g`: the gradients
    - `m`: the first moment
    - `v`: the second moment
    - `lr`: the learning rate
    - `iter`: the iteration number
    - `beta1`: the beta1 parameter
    - `beta2`: the beta2 parameter
    - `eps`: the epsilon value

    Returns:
    - the updated lr (also modifies m and v in place)
    """
    m[:] = beta1.*m + ((1-beta1).*g)
    v[:] = beta2.*v + ((1-beta2).*(g.^2))
    m_hat = m./(1-beta1^(iter))
    v_hat = v./(1-beta2^(iter))
    return lr.*(m_hat./(sqrt.(v_hat) .+ eps))
end


