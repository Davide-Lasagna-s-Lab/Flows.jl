using Documenter, Flows

makedocs(
    sitename = "Flows.jl",
    modules  = [Flows],
    authors  = "Davide Lasagna",
    pages    = [
        "Home" => "index.md",
        "Getting started" => [
            "Quick start"               => "quickstart.md",
        ],
        "Concepts" => [
            "Mathematical foundations"  => "foundations.md",
            "Architecture"              => "architecture.md",
        ],
        "Building blocks" => [
            "States and vector fields"  => "states.md",
            "Integration schemes"       => "schemes.md",
            "Time stepping"             => "time-stepping.md",
            "Coupled systems"           => "coupled.md",
            "Trajectory data"           => "trajectories.md",
            "Linearised dynamics"       => "linearised.md",
            "Symmetry transformations"  => "symmetry.md",
            "Quadrature equations"      => "quadrature.md",
        ],
        "Practice" => [
            "Cookbook"                  => "cookbook.md",
        ],
        "Reference" => [
            "Internals"                 => "internals.md",
            "API"                       => "api.md",
        ],
    ],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://davide-lasagna-s-lab.github.io/Flows.jl/stable/",
    ),
)

deploydocs(
    repo = "github.com/Davide-Lasagna-s-Lab/Flows.jl.git",
)
