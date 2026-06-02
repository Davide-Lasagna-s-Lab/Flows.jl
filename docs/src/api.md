# Full public API

This page collects the public docstrings exported by `Flows`. For
guided examples, start with the [Quick start](@ref) and the manual
pages; this reference is intentionally terse.

## Flow operator

The basic building block of this package is the [`Flow`](@ref) object,
a discrete approximation of the flow of a dynamical system. The
factory function [`flow`](@ref) is the public construction entry
point and has many overloads to cover every supported combination of
explicit / IMEX, single-state / coupled, default / custom call
dependency.

```@docs
flow
```

Objects of type [`Flow`](@ref) are callable; the call syntax is
documented on the type itself.

```@docs
Flows.Flow
Flows.InvalidSpanError
```

## Coupled states

```@docs
Coupled
couple
couplecopy
Flows.SymTransform
Flows.CoupledTransform
```

## Monitors and storages

```@docs
Monitor
reset!
times
samples
RAMStorage
period
isperiodic
timespan
storelast
degree
StoreNFromLast
```

## Stage caches

```@docs
AbstractStageCache
RAMStageCache
```

## Integration methods

```@docs
RK4
CNRK2
CB3R2R2
CB3R2R3c
CB3R2R3e
CB4R3R4
ImcA!
```

## Time stepping

```@docs
TimeStepConstant
TimeStepFromStorage
TimeStepFromCache
AbstractTimeStepFromHook
```

## Call dependencies

```@docs
CallDependency
```

## Standalone quadrature rules

```@docs
Flows.trapz
Flows.simps
```
