using Documenter, Flows

makedocs(sitename = "Flows.jl",
         modules  = [Flows],
         authors  = "Davide Lasagna",
         pages    = ["Home"                  => "index.md",
                     "Quick Start"           => "quickstart.md",
                     "Monitors"              => "monitors.md",
                     "Available methods"     => "available-methods.md",
                     "Time stepping schemes" => "time-stepping.md",
                     "Coupled systems"       => "coupled.md",
                     "Solution storages"     => "storage.md",
                     "Quadrature equations"  => "quadrature.md",
                     "Examples"              => "examples.md",
                     "Advanced features"     => "advanced.md",
                     "Full API"              => "api.md"])

deploydocs(
    repo = "github.com/gasagna/Flows.jl.git",
)
