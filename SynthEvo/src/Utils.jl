function save_ga_state(path, C, state, gd_options, ga_options, gd_perturbation_options, ga_perturbation_options)
    """
    Save the GA state to a file.

    Args:
    - `path`: the path to the file
    - `C`: the CRN
    - `state`: the GA state
    - `gd_options`: the gradient descent options
    - `ga_options`: the genetic algorithm options
    - `gd_perturbation_options`: the gradient descent perturbation options
    - `ga_perturbation_options`: the genetic algorithm perturbation options
    """
    save_object(path, Dict(
        "C" => C,
        "state" => state,
        "gd_options" => gd_options,
        "ga_options" => ga_options,
        "gd_perturbation_options" => gd_perturbation_options,
        "ga_perturbation_options" => ga_perturbation_options
    ))

    return nothing
end

function load_ga_state(path)
    """
    Load the GA state from a file.

    Args:
    - `path`: the path to the file

    Returns:
    - the GA state 
    (A dictionary with keys "C", "state", "gd_options", "ga_options", "gd_perturbation_options", "ga_perturbation_options")
    """
    return load_object(path)
end

function sample_parameters_from_LHC(number_of_sets, number_of_parameters, gens)
    """
    Sample parameters from a Latin Hypercube.

    Args:
    - `number_of_sets`: the number of sets to sample
    - `number_of_parameters`: the number of parameters to sample
    - `gens`: the number of generations (LHC hyperparameter)

    Returns:
    - a list of parameter sets sampled from the LHC
    """
    plan, _ = LatinHypercubeSampling.LHCoptim(number_of_sets, number_of_parameters, gens)
    params = LatinHypercubeSampling.scaleLHC(plan, [(0, 1) for _ in 1:number_of_parameters])
    return [params[i,:] for i in 1:number_of_sets]
end