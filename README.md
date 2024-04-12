# SynthEvo

A tool for finding CRN topologies under user-defined constraint

### Installation

To install the package, run the following command in `julia`:

```julia
using Pkg; Pkg.add("SynthEvo")
```

finally import the package in your code with:

```julia
using SynthEvo
```

### Examples

check the `examples` directory for some to see some commented use-cases of the package.

#### Use from source

To use this package directly from source, clone the repository and in `julia` run the following commands:

```julia
using Pkg; activate("/path/to/SynthEvo")
```

Where `/path/to/SynthEvo` is the path to the SynthEvo directory inside the cloned repository.

In this way, you can use the package for the current session. 

### Changelog

v0.0.0: Initial release with basic functionality

v0.1.0: refactoring and optimization

* Updataed code to imporve the performance of the symbolic computation
* Added notebooks with examples
* Added function documentation 



