

# is_updated means that the fitness value is up to date

function initialize_state(np, ga_options)
    """
    Initialize a pool as defined in ga_options.
    Each parameter vector will have size np.

    Returns a state object.

    Args:
    - np: number of parameters
    - ga_options: a struct with at least the following fields:
        - genetic_pool_size: number of parameter vectors

    Returns:
    - state: a struct with the following fields:
        - pool: a list of parameter vectors
        - is_updated: a list of booleans
        - fitness: a list of floats
        - history: a struct with the following fields:
            - best_loss: a list of floats
            - mean_loss: a list of floats
    """
    parameter_pool = [[rand() for _ in 1:np] for _ in 1:ga_options.genetic_pool_size]
    is_updated = [false for _ in 1:ga_options.genetic_pool_size]
    fitness = [0. for _ in 1:ga_options.genetic_pool_size]
    state = (
        pool = parameter_pool,
        is_updated = is_updated,
        fitness = fitness,
        history = (best_loss = [], mean_loss = [])
    )
    return state
end


function prepare_GA_loss(C, base_problem, ga_perturbation_options)
    """
    Prepare a function that computes the loss of a state.

    Args:
    - C: a CRN object
    - base_problem: the ODEproblem to solve 
    - ga_perturbation_options: a struct with the following fields:
        - use_random_perturbation: a boolean
        - input: a list of perturbation events
        - perturbation_list: a list of perturbation events
        - t0: a float
        - t1: a float

    Returns:
    A function that takes a state and returns the computed loss for each iof them
    """
    function compute_symbolic_loss(state)

        to_compute = [i for i in 1:length(state.pool) if !state.is_updated[i]]
        if ga_perturbation_options.use_random_perturbation
            perturbation_list = perturbation_events(ga_perturbation_options.input, ga_perturbation_options.perturbation_list) # one sample per IT
        else
            perturbation_list = ga_perturbation_options.perturbation_list 
        end

        trajectories = runp_SF_CPU(C, base_problem, state.pool[to_compute], ga_perturbation_options.t0, ga_perturbation_options.t1, perturbation_list)

        trj_index = 1
        Threads.@threads for i in to_compute
            state.fitness[i] = 0 
            for j in 1:length(trajectories[trj_index][2])
                trajectory = merge_solutions(trajectories[trj_index][1], trajectories[trj_index][2], j, ga_perturbation_options.t0)
                state.fitness[i] += eval_loss(C, ga_perturbation_options.loss_fun, state.pool[i], trajectory, false).loss
                state.is_updated[i] = true
            end
            state.fitness[i] /= length(trajectories[trj_index][2])
            trj_index += 1
        end

        return nothing
    end
    return compute_symbolic_loss
end


function symbolic_evolve(loss_function_ga, state, follow_gd_fun, ga_options)
    """
    Evolve the genetic pool according to the genetic algorithm options.

    Args:
    - loss_function_ga: a function that computes the loss of a state
    - state: a struct with the following fields:
        - pool: a list of parameter vectors
        - is_updated: a list of booleans
        - fitness: a list of floats
        - history: a struct with the following fields:
            - best_loss: a list of floats
            - mean_loss: a list of floats
    - follow_gd_fun: a function that updates the parameter vector according to the gradient descent algorithm
    - ga_options: a struct with the following fields:
        - dp: a float
        - elite: an integer
        - worst: an integer
        - mutation_rate: a float
        - gradient_mutation_rate: a float
        - death_rate: a float
        - duplication_rate: a float
        - crossover_rate: a float
        - p_cross: a float
    
    Returns:
    - state: the evolved state

    """
    np = length(state.pool[1])

    # update losses
    loss_function_ga(state)

    ranking = sortperm(state.fitness)
    elite_pool = ranking[1:ga_options.elite]
    worst_pool = ranking[end-ga_options.worst:end]
    others_pool = ranking[ga_options.elite+1:end-ga_options.worst]

    push!(state.history.best_loss, state.fitness[elite_pool[1]])
    push!(state.history.mean_loss, mean(state.fitness))

    function perturb(dp, np)
        return dp .* rand(np) .- (dp/2)
    end
    
    # kill the unfit and replace them with the elite's offsprings
    Threads.@threads for i in worst_pool
        other = rand(1:ga_options.elite)
        state.pool[i][:] .= max.(0, state.pool[elite_pool[other]] + perturb(ga_options.dp, np))
        state.is_updated[i] = false
    end
    # spare the elite
    Threads.@threads for i in elite_pool
        p = rand()
        if p < ga_options.gradient_mutation_rate
            state.pool[i][:] .= follow_gd_fun(state.pool[i])
            state.is_updated[i] = true # now the loss is deterministic, we can avoid re-evaluating
        end
    end
    # mutate the rest
    for i in others_pool
        p = rand()

        # TODO this part could be improved if we associate an action and a probability to each operation

        if p < ga_options.mutation_rate
            state.pool[i][:] .= max.(0, perturb(ga_options.dp, np) + state.pool[i])
            state.is_updated[i] = false
        elseif p < ga_options.gradient_mutation_rate + ga_options.mutation_rate
            state.pool[i][:] .= follow_gd_fun(state.pool[i])
        elseif p < ga_options.death_rate + ga_options.gradient_mutation_rate + ga_options.mutation_rate
            state.pool[i][:] .= [rand() for _ in 1:np]
            state.is_updated[i] = false
        elseif p < ga_options.duplication_rate + ga_options.death_rate + ga_options.gradient_mutation_rate + ga_options.mutation_rate
            if i+1 > length(others_pool)
                state.is_updated[i] = true
                continue
            end
            clone_destination = rand(i+1:length(others_pool)) # replace only worse outcomes
            state.pool[others_pool[clone_destination]][:] .= state.pool[i]
            state.fitness[others_pool[clone_destination]] = state.fitness[i]
            state.is_updated[others_pool[clone_destination]] = state.is_updated[i]
        elseif p < ga_options.crossover_rate + ga_options.duplication_rate + ga_options.death_rate + ga_options.gradient_mutation_rate + ga_options.mutation_rate
            crossover_mate = rand(1:length(others_pool))
            crossover_mask = rand(np) .> ga_options.p_cross
            state.pool[i][:] .= ((1 .- crossover_mask)*state.pool[i]) + state.pool[others_pool[crossover_mate]]*crossover_mask
            state.is_updated[i] = false
        end
    end
    return state
end


### 

# in this 
function symbolic_evolve_NFB(loss_function_ga, state, follow_gd_fun, ga_options, output_degradation_parameter, minimal_degradation_rate)
    # requires functions for the rates of mutation, gradient mutation, death, duplication, crossover
    
    np = length(state.pool[1])
    Threads.@threads for i in 1:length(state.pool)
        state.pool[i][output_degradation_parameter] = max.(minimal_degradation_rate, state.pool[i][output_degradation_parameter])
    end

    loss_function_ga(state)

    ranked_pool = sortperm(state.fitness)

    push!(state.history.best_loss, state.fitness[ranked_pool[1]])
    push!(state.history.mean_loss, mean(state.fitness))

    function perturb(dp, np)
        return dp .* rand(np) .- (dp/2)
    end

    function perturb1(dp, np)
        return dp .* rand() .- (dp/2)
    end
    
    for i in ranked_pool
        p = rand()

        current_mutation_rate = ga_options.mutation_rate(i/length(ranked_pool))
        current_gradient_mutation_rate = ga_options.gradient_mutation_rate(i/length(ranked_pool)) 
        current_death_rate = ga_options.death_rate(i/length(ranked_pool)) 
        current_duplication_rate = ga_options.duplication_rate(i/length(ranked_pool)) 
        current_crossover_rate = ga_options.crossover_rate(i/length(ranked_pool))

        # mutation
        if 0 < p < current_mutation_rate
            state.pool[i][output_degradation_parameter] = max(0, perturb1(ga_options.dp, np) + state.pool[i][output_degradation_parameter])
            state.is_updated[i] = false
        # gradient mutation
        elseif p < current_gradient_mutation_rate + current_mutation_rate
            state.pool[i][:] .= follow_gd_fun(state.pool[i])
        # death
        elseif p < current_death_rate + current_gradient_mutation_rate + current_mutation_rate
            state.pool[i][:] .= [rand() for _ in 1:np]
            state.is_updated[i] = false
        # duplication
        elseif p < current_duplication_rate + current_death_rate + current_gradient_mutation_rate + current_mutation_rate
            if i+1 > length(ranked_pool)
                state.is_updated[i] = true
                continue
            end
            clone_destination = rand(i+1:length(ranked_pool)) # replace only worse outcomes
            state.pool[ranked_pool[clone_destination]][:] .= state.pool[i]
            state.fitness[ranked_pool[clone_destination]] = state.fitness[i]
            state.pool[ranked_pool[clone_destination]][:] .= max.(0, perturb(ga_options.dp*2, np) + state.pool[i]) # strong mutation
            state.is_updated[ranked_pool[clone_destination]] = false
        # crossover
        elseif p < current_crossover_rate + current_duplication_rate + current_death_rate + current_gradient_mutation_rate + current_mutation_rate
            # by now unused
            crossover_mate = rand(1:length(ranked_pool))
            crossover_mask = rand(np) .> ga_options.p_cross
            state.pool[i][:] .= ((1 .- crossover_mask)*state.pool[i]) + state.pool[ranked_pool[crossover_mate]]*crossover_mask
            state.is_updated[i] = false
        end
    end
    return state
end