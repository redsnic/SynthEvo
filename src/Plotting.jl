
function plot_history(state)
"""
Plot the history of the GA state.

Args:
- `state`: the GA state (.history field with best_loss and mean_loss required)
"""
    plot(state.history.best_loss, label="best loss", xlabel="generation", ylabel="loss", title="Best loss vs generation", lw=2, legend=:topright)
    plot!(state.history.mean_loss, label="mean loss", lw=2, line=:dash)
end

function quick_trajectory_plot(C, p0, input0, input1, t0, t1, target, u0=nothing, prec=10)
    """
    Quickly plot a trajectory of the CRN.

    Args:
    - `C`: the CRN
    - `p0`: the parameters
    - `input0`: the initial input
    - `input1`: the final input
    - `t0`: the initial time
    - `t1`: the final time
    - `prec`: the precision of the plot
    
    Returns:
    - the plot
    """
    if u0 == nothing
        u0 = [0 for _ in 1:C.N]
    end
    base_problem = make_base_problem(C, C.ode, u0, [input0], p0, t0, true, 1/prec)
    shared_solution, solutions = run_SF_CPU(C, base_problem, t0, t1, input1, 1e-5, 1e-5, true, 0.25)
    for i in 1:length(solutions)
        sol = SynthEvo.merge_solutions(shared_solution, solutions, i, t0)
        if i == 1
            plot( [(t/prec) for t in 0:t1*prec], vec2mat([sol(t/prec)[1:C.N] for t in 0:t1*prec]), legend=:topleft, xlabel="time", ylabel="concentration", title="Trajectory")
        else
            plot!( [(t/prec) for t in 0:t1*prec], [sol(t/prec)[target] for t in 0:t1*prec], color=:black, linestyle=:dash, alpha=0.2, label=false)
        end
    end
    return plot!()
end

function quick_sensitivity_plot(C, p0, input0, input1, t0, t1, target, u0=nothing, prec=10)
    """
    Quickly plot a sensitivities of the CRN.

    Args:
    - `C`: the CRN
    - `p0`: the parameters
    - `input0`: the initial input
    - `input1`: the final input
    - `t0`: the initial time
    - `t1`: the final time
    - `prec`: the precision of the plot
    
    Returns:
    - the plot
    """
    if u0 == nothing
        u0 = [0 for _ in 1:C.N]
    end
    base_problem = make_base_problem(C, C.ode, u0, [input0], p0, t0, true, 1/prec)
    shared_solution, solutions = run_SF_CPU(C, base_problem, t0, t1, input1, 1e-5, 1e-5, true, 0.25)
    sol = merge_solutions(shared_solution, solutions, 1, t0)
    return plot([t/prec for t in 0:t1*prec], vec2mat([sol(t/prec)[C.N+length(C.control)+1:end] for t in 0:t1*prec]), xlabel="time", ylabel="sensitivity", title="Sensitivities", label=false)
end

### symGD

function symGD_plot_loss(results)
    """
    Plot the loss of the symGD optimization.

    Args:
    - `results`: the results of the optimization (symbolic gradient descent)
    """
    plot(results.loss_tape, label="loss", xlabel="iteration", ylabel="loss", title="Loss vs iteration", lw=2, legend=:topright)
end

function symGD_plot_parameters(results)
    """
    Plot the parameters of the symGD optimization.

    Args:
    - `results`: the results of the optimization (symbolic gradient descent)
    """
    plot(0:length(results.loss_tape), vec2mat(results.parameter_tape), label=false, xlabel="iteration", ylabel="parameter value", title="Parameters vs iteration", lw=2, legend=:topright)
end

function symGD_plot_gradient(results)
    """
    Plot the gradient of the symGD optimization.

    Args:
    - `results`: the results of the optimization (symbolic gradient descent)
    """
    plot(results.gradient_tape, label=false, xlabel="iteration", ylabel="gradient", title="Gradient vs iteration", lw=2, legend=:topright)
end

