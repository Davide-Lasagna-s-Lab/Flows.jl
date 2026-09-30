using Documenter, Flows

makedocs(
    sitename = "Flows.jl",
    modules  = [Flows],
    authors  = "Davide Lasagna",
    # `repo` wires the "Edit on GitHub" / repository link in the page header.
    # Documenter picks up `docs/src/assets/logo.svg` automatically and renders
    # it in the top-left corner.
    repo     = Remotes.GitHub("Davide-Lasagna-s-Lab", "Flows.jl"),
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
    # Cross-reference / docstring checks are advisory rather than fatal so a
    # missing `@ref` does not break the deploy.
    warnonly  = true,
    checkdocs = :none,
)

deploydocs(
    repo      = "github.com/Davide-Lasagna-s-Lab/Flows.jl.git",
    devbranch = "master",
    push_preview = false,
)
