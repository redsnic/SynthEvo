using Combinatorics

mutable struct FullyConnectedCRN <: CRN
    N::Int
    time::Num
    species::Symbolics.Arr{Num, 1}
    control::Num
    parameters::Symbolics.Arr{Num, 1}
    number_of_parameters::Int
    sensitivity_variables::Symbolics.Arr{Num, 2}
    crn::Union{Nothing, ReactionSystem}
    ode::Union{Nothing, ODESystem}
    ext_ode::Union{Nothing, ODESystem}
    to_visible_order_indexes::Union{Nothing,Vector{Int64}}
    to_hidden_order_indexes::Union{Nothing,Vector{Int64}}
    to_visible_order::Union{Nothing,Function}
    to_hidden_order::Union{Nothing,Function}
end

function reorder2visible(C, par_v)
    dict = Dict()
    for i in 1:C.number_of_parameters
        dict[ModelingToolkit.parameters(C.ext_ode)[i]] = par_v[i]
    end
    out = []
    for i in 1:C.number_of_parameters
        push!(out, dict[C.parameters[i]])
    end
    return out
end

function reorder2hidden(visible_order)
    out = zeros(length(visible_order))
    for i in 1:length(visible_order)
        out[visible_order[i]] = i
    end
    return out
end


function make_FullyConnectedNonExplosive_CRN(N::Int)
    """
    Create a fully connected CRN model

    Args:
    - N: number of species

    Returns:
    - the fully connected CRN model
    """
    @variables t
    @species x(t)[1:N]
    @variables U(t)

    np = count_parameters_FullyConnectedNonExplosive(N)
    @parameters k[1:np]
    @variables ks(t)[1:N, 1:np]
    
    C = FullyConnectedCRN(N, t, x, U, k, np, ks, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    C.crn = create_reactions(C)
    C.ode = convert(ODESystem, C.crn)
    C.ode = add_control_equations(C, [U])
    C.ext_ode = make_sensitivity_ode(C)
 
    C.to_visible_order_indexes = reorder2visible(C, 1:np)
    C.to_hidden_order_indexes = reorder2hidden(C.to_visible_order_indexes)

    C.to_visible_order = (p) -> p[C.to_visible_order_indexes]
    C.to_hidden_order = (p) -> p[C.to_hidden_order_indexes]

    return C
end

# --- Getters --- 

function species(C::FullyConnectedCRN)
    return C.species
end

function control(C::FullyConnectedCRN)
    return C.control
end

function parameters(C::FullyConnectedCRN)
    return C.parameters
end

function number_of_parameters(C::FullyConnectedCRN)
    return C.number_of_parameters
end

function sensitivity_variables(C::FullyConnectedCRN)
    return C.sensitivity_variables
end

function time(C::FullyConnectedCRN)
    return C.time
end

function n_species(C::FullyConnectedCRN)
    return C.N
end

function count_parameters_FullyConnectedNonExplosive(N::Int)
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



function create_reactions(C::FullyConnectedCRN)
    """
    Use Catalyst inteface to create the reactions of the fully-connected CRN model 
    with N species

    Args:
    - N: number of species

    Returns:
    - reactions: the list of reactions
    """
    N = C.N
    t = C.time
    x = C.species
    U = C.control
    par_k = C.parameters
    np = C.number_of_parameters

    reactions = []
    global_counter = 1

    push!(reactions, Reaction(U, nothing, [x[1]]))
    #0 -> 1
    for i in 2:N
        push!(reactions, Reaction(par_k[global_counter], nothing, [x[i]]))
        global_counter += 1
    end
    # 0 -> 2
    for (i,j) in combinations(1:N, 2)
        if i != j
            push!(reactions, Reaction(par_k[global_counter], nothing, [x[i], x[j]]))
        else
            push!(reactions, Reaction(par_k[global_counter], nothing, [x[i]], nothing, [2]))
        end
        global_counter += 1
    end
    # # # 1 -> 1
    for i in 1:N
        for j in 1:N
            if i != j
                push!(reactions, Reaction(par_k[global_counter], [x[i]], [x[j]]) )
                global_counter += 1
            end
        end
    end
    # # 1 -> 2
    for i in 1:N
        for (j,k) in with_replacement_combinations(1:N, 2)
            if !(i == j == k)
                if j != k
                    push!(reactions, Reaction(par_k[global_counter], [x[i]], [x[j], x[k]]))
                else
                    push!(reactions, Reaction(par_k[global_counter], [x[i]], [x[j]], [1], [2]))
                end
                global_counter += 1
            end
        end
    end
    # # # 1 -> 0
    for i in 1:N
        push!(reactions, Reaction(par_k[global_counter], [x[i]], nothing))
        global_counter += 1
    end
    # # # 2 -> 0
    for (i,j) in with_replacement_combinations(1:N, 2)
        if i != j
            push!(reactions, Reaction(par_k[global_counter], [x[i], x[j]], nothing))
        else
            push!(reactions, Reaction(par_k[global_counter], [x[i]], nothing, [2], nothing))
        end
        global_counter += 1
    end
    # # # 2 -> 1
    for (i,j) in with_replacement_combinations(1:N, 2)
        for k in 1:N
            if !(i == j == k)
                if i != j
                    push!(reactions, Reaction(par_k[global_counter], [x[i], x[j]], [x[k]]))
                else
                    push!(reactions, Reaction(par_k[global_counter], [x[i]], [x[k]], [2], [1]))
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
                    push!(reactions, Reaction(par_k[global_counter], [x[i], x[j]], [x[k], x[l]]))
                elseif i == j && k != l
                    push!(reactions, Reaction(par_k[global_counter], [x[i]], [x[k], x[l]], [2], [1,1]))
                elseif i != j && k == l
                    push!(reactions, Reaction(par_k[global_counter], [x[i], x[j]], [x[k]], [1,1], [2]))
                else
                    push!(reactions, Reaction(par_k[global_counter], [x[i]], [x[k]], [2], [2]))
                end
                global_counter += 1
            end
        end
    end
    @named crn = ReactionSystem(reactions, t)

    return crn
end


