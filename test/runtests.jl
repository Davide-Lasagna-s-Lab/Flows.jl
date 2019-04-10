using Test
using Flows

include("test_timestepping.jl")
include("test_tableaux.jl")
include("test_losslessrange.jl")
include("test_imca.jl")
include("test_steps.jl")
include("test_integrator.jl")
include("test_quadrature.jl")
include("test_monitor.jl")
include("test_storage.jl")
include("test_stagecache.jl")
include("test_linearisation.jl")
include("test_analytic_adjoint.jl")
include("test_gradient.jl")