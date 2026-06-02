# Full public API

Public docstrings for every exported symbol, grouped by topic. For guidance on which symbols to reach for in which situation, start with the manual pages — this reference is intentionally terse.

## Flow operator

```@docs
flow
Flows.Flow
Flows.InvalidSpanError
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
ImcA!
```

## Time stepping

```@docs
TimeStepConstant
TimeStepFromStorage
Flows.TimeStepFromCache
Flows.AbstractTimeStepFromHook
```

## Monitors

```@docs
Monitor
reset!
times
samples
StoreNFromLast
```

## Storages

```@docs
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
