"""
    Flows.jl

A Julia package to define and manipulate flow operators of dynamical systems.
"""
module Flows

include("couple.jl")
include("tableaux.jl")
include("stagecache.jl")
include("system.jl")
include("storage.jl")

include("steps/shared.jl")
include("steps/rk4.jl")
include("steps/CNRK2.jl")
include("steps/CB3R2R.jl")
include("steps/CB4R3R.jl")

include("timestepping.jl")
include("logger.jl")
include("monitor.jl")
include("storenfromlast.jl")
include("imca.jl")
include("stepper.jl")
include("integrator.jl")
include("utils.jl")

end