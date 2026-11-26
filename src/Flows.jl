"""
    Flows.jl

A Julia package to define and manipulate flow operators of dynamical systems.
"""
module Flows

using LinearAlgebra

import Printf: @sprintf, Format, format

export Coupled, couple, couplecopy
export AbstractStageCache, RAMStageCache
export CallDependency
export AbstractStorage, RAMStorage, times, samples, degree, storelast, timespan, period, isperiodic
export StoreNFromLast
export AbstractTimeStepping,
       AbstractTimeStepFromHook,
       TimeStepConstant,
       TimeStepFromCache,
       TimeStepFromStorage
export Monitor, reset!
export ImcA!
export flow, InvalidSpanError
export simps, trapz

export RK4, CNRK2, CB3R2R2, CB3R2R3e, CB3R2R3c, CB4R3R4

include("couple.jl")
include("tableaux.jl")
include("stagecache.jl")
include("system.jl")
include("storage.jl")

include("steps/shared.jl")
include("steps/rk4.jl")
include("steps/rk8.jl")
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
