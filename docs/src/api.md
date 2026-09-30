# Full public API

Docstrings for every exported and documented internal symbol, grouped by topic. For guidance on which symbol to reach for in which situation, start with the manual pages — this reference is intentionally terse.

## Flow operator

```@docs
flow
Flows.Flow
Flows.InvalidSpanError
```

## System wrapper

```@docs
Flows.System
```

## States and coupled states

```@docs
Coupled
couple
couplecopy
Flows.SymTransform
Flows.CoupledTransform
```

## Call dependencies

```@docs
CallDependency
```

## Integration schemes

```@docs
RK4
CNRK2
CB3R2R2
CB3R2R3c
CB3R2R3e
CB4R3R4
```

### Method base type and integration modes

```@docs
Flows.AbstractMethod
Flows.AbstractMode
Flows.NormalMode
Flows.ContinuousMode
Flows.DiscreteMode
```

### IMEX implicit solves

```@docs
ImcA!
Flows.ImcA_mul!
```

## Time stepping

```@docs
TimeStepConstant
TimeStepFromStorage
Flows.TimeStepFromCache
Flows.AbstractTimeStepFromHook
Flows.Steps
```

## Monitors

```@docs
Monitor
reset!
times
samples
StoreNFromLast
Flows.Logger
```

## Storages

```@docs
Flows.AbstractStorage
RAMStorage
period
isperiodic
timespan
storelast
degree
```

## Stage caches

```@docs
AbstractStageCache
RAMStageCache
```

## Standalone quadrature rules

```@docs
Flows.trapz
Flows.simps
```
