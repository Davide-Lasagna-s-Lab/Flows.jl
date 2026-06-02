# Flows.jl

```@raw html
<p align="center">
  <img src="assets/logo.svg" alt="Flows.jl logo" width="640">
</p>
```

A flow-like API to solve differential equations.

## Rationale
Many numerical algorithms in dynamical systems theory require the action of a flow operator associated with a dynamical system, rather than a solve-an-initial-value-problem-and-store-its-solution approach. This package provides an API geared toward those needs, without the ambition to be a general-purpose ODE package.

## Installation
This package is not registered in Julia's `General` registry but can be installed directly from GitHub. In the Julia REPL hit `]` to enter package mode, then type
```julia
add https://github.com/gasagna/Flows.jl
```

## Table of Contents
```@contents
Pages = ["quickstart.md",
         "monitors.md",
         "available-methods.md",
         "time-stepping.md",
         "coupled.md",
         "storage.md",
         "quadrature.md",
         "examples.md",
         "advanced.md",
         "api.md"]
Depth = 1
```
