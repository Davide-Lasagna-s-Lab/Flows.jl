export available_schemes, imex_schemes

"""
    imex_schemes

The implicit–explicit schemes of the package by name: a dictionary from the symbol of a scheme
to its constructor, so that `imex_schemes[:CB3R2R3e](x)` is `CB3R2R3e(x)`. They take the linear
part of a system implicitly, by [`ImcA!`](@ref): `:CNRK2`, `:CB3R2R2`, `:CB3R2R3e`, `:CB3R2R3c`
and `:CB4R3R4`. A package that lets its user name the scheme checks the name against it.
"""
const imex_schemes = Dict{Symbol, Any}(:CNRK2    => CNRK2,
                                       :CB3R2R2  => CB3R2R2,
                                       :CB3R2R3e => CB3R2R3e,
                                       :CB3R2R3c => CB3R2R3c,
                                       :CB4R3R4  => CB4R3R4)

"""
    available_schemes

All the schemes of the package by name, a dictionary from the symbol of a scheme to its
constructor, as [`imex_schemes`](@ref): the implicit–explicit ones, and the explicit ones,
`:RK4` and `:RK8`.
"""
const available_schemes = merge(imex_schemes, Dict{Symbol, Any}(:RK4 => RK4,
                                                                 :RK8 => RK8))
