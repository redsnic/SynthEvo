
using Statistics

function evolve(crn, prob, s_loss, callback, state, tspan, dp, genetic_pool_size, elite, worst, death_rate, mutation_rate, gradient_mutation_rate, follow_gd_fun, duplication_rate, crossover_rate, p_cross, loss_rep)
    Threads.@threads for i in 1:length(state.pool)
        if is_updated[i]
            continue
        end
        # parameter_pool[i].p[:U] = 1. # reset base input to 1 (it cannot be lerned)
        # #current_problem = ODEProblem(crn, parameter_pool[i].u0, tspan, parameter_pool[i].p) #remake(prob; u0 = params.u0, p = params.p)
        # current_problem = remake(prob; u0 = state.pool[i].u0, p = state.pool[i].p)
        # state.fitness[i] = 0.
        # for _ in 1:loss_rep
        #     sol = integrate(current_problem, tspan, callback)
        #     state.fitness[i] += full_loss(state.pool[i].p, sol, N, tspan[2]/2, tspan[2])
        # end
        # state.fitness[i] /= loss_rep
        # state.is_updated[i] = true
        state.fitness[i] = s_loss([x for x in values(state.pool[i].p)])
        state.is_updated[i] = true
    end
    ranking = sortperm(state.fitness)
    elite_pool = ranking[1:elite]
    worst_pool = ranking[end-worst:end]
    others_pool = ranking[elite+1:end-worst]

    push!(state.history.best_loss, state.fitness[elite_pool[1]])
    push!(state.history.mean_loss, mean(state.fitness))

    function perturb!(dp)
        return dp * rand() - dp/2
    end

    # spare the elite
    Threads.@threads for i in elite_pool
        p = rand()
        if p < gradient_mutation_rate
            #current_problem = ODEProblem(crn, parameter_pool[i].u0, tspan, parameter_pool[i].p)
            #current_problem = remake(prob; u0 = state.pool[i].u0, p = state.pool[i].p)
            new_p = assemble_opt_parameters_and_varables(follow_gd_fun([x for x in values(state.pool[i].p)]), N)
            for j in keys(state.pool[i].p)
                state.pool[i].p[j] = new_p.p[j]
            end
            state.is_updated[i] = false  # re-evaluate, maybe we got lucky with the loss...
        end
    end
    # kill the unfit and replace them with the elite's offsprings
    Threads.@threads for i in worst_pool
        other = rand(1:elite)
        for j in keys(state.pool[i].p)
            state.pool[i].p[j] = max(0, state.pool[elite_pool[other]].p[j] + perturb!(dp))
        end
        state.is_updated[i] = false
    end
    # mutate the rest
    for i in others_pool
        p = rand()
        if p < mutation_rate
            map!(x -> max(0, perturb!(dp) + x), values(state.pool[i].p))
            state.is_updated[i] = false
        elseif p < gradient_mutation_rate
            state.pool[i].p[:] = follow_gd_fun(state.pool[i].p)
        elseif p < death_rate
            state.pool[i] = assemble_opt_parameters_and_varables([rand() for _ in 1:np], N)
            state.is_updated[i] = false
        elseif p < duplication_rate
            if i+1 > length(others_pool)
                state.is_updated[i] = false
                continue
            end
            clone_detination = rand(i+1:length(others_pool)) # replace only worse outcomes
            state.pool[others_pool[clone_detination]] = state.pool[i]
            state.fitness[others_pool[clone_detination]] = state.fitness[i]
            state.is_updated[others_pool[clone_detination]] = state.is_updated[i]
        elseif p < crossover_rate
            crossover_mate = rand(1:length(others_pool))
            map!((x,y) -> rand() > p_cross ? (x+y)/2 : x, values(state.pool[i].p), values(state.pool[others_pool[crossover_mate]].p))
            state.is_updated[i] = false
        end
    end
    return state
end

# is_updated means that the fitness value is up to date
function symbolic_evolve(crn, loss_function, state, dp, genetic_pool_size, elite, worst, death_rate, mutation_rate, gradient_mutation_rate, follow_gd_fun, duplication_rate, crossover_rate, p_cross)
    np = length(state.pool[1])
    Threads.@threads for i in 1:length(state.pool)
        if is_updated[i]
            continue
        end
        state.fitness[i] = loss_function(state.pool[i])
        state.is_updated[i] = true
    end
    ranking = sortperm(state.fitness)
    elite_pool = ranking[1:elite]
    worst_pool = ranking[end-worst:end]
    others_pool = ranking[elite+1:end-worst]

    push!(state.history.best_loss, state.fitness[elite_pool[1]])
    push!(state.history.mean_loss, mean(state.fitness))

    function perturb(dp, np)
        return dp .* rand(np) .- (dp/2)
    end
    
    # kill the unfit and replace them with the elite's offsprings
    Threads.@threads for i in worst_pool
        other = rand(1:elite)
        state.pool[i][:] .= max.(0, state.pool[elite_pool[other]] + perturb(dp, np))
        state.is_updated[i] = false
    end
    # spare the elite
    Threads.@threads for i in elite_pool
        p = rand()
        if p < gradient_mutation_rate
            state.pool[i][:] .= follow_gd_fun(state.pool[i])
            state.is_updated[i] = true # now the loss is deterministic, we can avoid re-evaluating
        end
    end
    # mutate the rest
    for i in others_pool
        p = rand()
        if p < mutation_rate
            state.pool[i][:] .= max.(0, perturb(dp, np) + state.pool[i])
            state.is_updated[i] = false
        elseif p < gradient_mutation_rate
            state.pool[i][:] .= follow_gd_fun(state.pool[i])
        elseif p < death_rate
            state.pool[i][:] .= [rand() for _ in 1:np]
            state.is_updated[i] = false
        elseif p < duplication_rate
            if i+1 > length(others_pool)
                state.is_updated[i] = true
                continue
            end
            clone_detination = rand(i+1:length(others_pool)) # replace only worse outcomes
            state.pool[others_pool[clone_detination]][:] .= state.pool[i]
            state.fitness[others_pool[clone_detination]][:] .= state.fitness[i]
            state.is_updated[others_pool[clone_detination]][:] .= state.is_updated[i]
        elseif p < crossover_rate
            crossover_mate = rand(1:length(others_pool))
            crossover_mask = rand(np) .> p_cross
            state.pool[i][:] .= ((1 .- crossover_mask)*state.pool[i]) + state.pool[others_pool[crossover_mate]]*crossover_mask
            state.is_updated[i] = false
        end
    end
    return state
end


### 

# in this 
function symbolic_evolve_NFB(crn, loss_function, state, dp, genetic_pool_size, death_rate, mutation_rate, gradient_mutation_rate, follow_gd_fun, duplication_rate, crossover_rate, p_cross, output_degradation_parameter, minimal_degradation_rate)
    np = length(state.pool[1])
    Threads.@threads for i in 1:length(state.pool)
        if is_updated[i]
            continue
        end
        # keep output degradation active
        state.pool[i][output_degradation_parameter] = max.(minimal_degradation_rate, state.pool[i][output_degradation_parameter])
        state.fitness[i] = loss_function(state.pool[i])
        state.is_updated[i] = true
    end

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

        current_mutation_rate = mutation_rate(i/length(ranked_pool))
        current_gradient_mutation_rate = gradient_mutation_rate(i/length(ranked_pool)) + current_mutation_rate
        current_death_rate = death_rate(i/length(ranked_pool)) + current_gradient_mutation_rate
        current_duplication_rate = duplication_rate(i/length(ranked_pool)) + current_death_rate

        # mutation
        if 0 < p < current_mutation_rate
            state.pool[i][output_degradation_parameter] = max(0, perturb1(dp, np) + state.pool[i][output_degradation_parameter])
            state.is_updated[i] = false
        # gradient mutation
        elseif current_mutation_rate < p < current_gradient_mutation_rate
            state.pool[i][:] .= follow_gd_fun(state.pool[i])
        # death
        elseif current_gradient_mutation_rate < p < current_death_rate
            state.pool[i][:] .= [rand() for _ in 1:np]
            state.is_updated[i] = false
        # duplication
        elseif current_death_rate < p < current_duplication_rate
            if i+1 > length(ranked_pool)
                state.is_updated[i] = true
                continue
            end
            clone_detination = rand(i+1:length(ranked_pool)) # replace only worse outcomes
            state.pool[ranked_pool[clone_detination]][:] .= state.pool[i]
            state.fitness[ranked_pool[clone_detination]] = state.fitness[i]
            state.pool[ranked_pool[clone_detination]][:] .= max.(0, perturb(dp*2, np) + state.pool[i]) # strong mutation
            state.is_updated[ranked_pool[clone_detination]] = false
        # crossover
        elseif p < crossover_rate(i/length(ranked_pool))
            # by now unused
            crossover_mate = rand(1:length(ranked_pool))
            crossover_mask = rand(np) .> p_cross
            state.pool[i][:] .= ((1 .- crossover_mask)*state.pool[i]) + state.pool[ranked_pool[crossover_mate]]*crossover_mask
            state.is_updated[i] = false
        end
    end
    return state
end