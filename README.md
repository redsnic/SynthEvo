# SynthEvo

A tool for finding CRN topologies under user-defined constraint

### Installation

To install the package, clone this GitHub repo and run the following command in `julia`:

```julia
using Pkg; Pkg.dev("</path/to/cloned/repo>/SynthEvo")
```

Replace the path between angular brackets with the one applying to your machine. The final
path will point to the `SynthEvo` directory **inside** the cloned repository.
Finally, import the package in your code with:

```julia
using SynthEvo
```

### Examples

check the `examples` directory for some to see some commented use-cases of the package.

#### Use from source

If you just want to test the package without installing it, clone the repository and in `julia` run the following commands:

```julia
using Pkg; activate("</path/to/cloned/repo>/SynthEvo")
```

Where `/path/to/SynthEvo` is the path to the SynthEvo directory inside the cloned repository.

In this way, you can use the package for the current session. 

### Changelog

v0.0.0: Initial release with basic functionality

v0.1.0: refactoring and optimization

* Updataed code to imporve the performance of the symbolic computation
* Added notebooks with examples
* Added function documentation 
