# ============================================================================
# Tableau types
# ============================================================================
#
# A `Tableau` is a Butcher tableau in the standard sense: the matrix `a`,
# the main weights `b`, the embedded weights `e` (used for error estimation
# in adaptive schemes), and the abscissae `c`. `IMEXTableau` bundles two
# such tableaux side by side — the implicit and explicit parts of an
# implicit-explicit Runge–Kutta scheme.

"""
    AbstractTableau{T, NS}

Supertype for Runge–Kutta tableaux. `T` is the coefficient element
type; `NS` is the number of internal stages.
"""
abstract type AbstractTableau{T, NS} end

"""
    nstages(::AbstractTableau) -> Int

Return the number of internal stages of the tableau.
"""
nstages(::AbstractTableau{T, NS}) where {T, NS} = NS

"""
    Tableau{T, NS}(a::Matrix, b::Vector, e::Vector, c::Vector)

Concrete Butcher tableau. The matrix `a` is `NS × NS`; the vectors
`b`, `e`, and `c` all have length `NS`. The constructor checks the
shape invariants.
"""
struct Tableau{T, NS} <: AbstractTableau{T, NS}
    a::Matrix{T}
    b::Vector{T}
    e::Vector{T}
    c::Vector{T}
    function Tableau{NS}(a::Matrix{T},
                         b::Vector{T},
                         e::Vector{T},
                         c::Vector{T}) where {T, NS}
        NS == size(a, 1) == size(a, 2) == length(b) == length(c) == length(e) ||
            error("invalid tableau input")
        new{T, NS}(a, b, e, c)
    end
end

"""
    Tableau(a, b, e, c) -> Tableau

Outer constructor that promotes the element types of the four
arrays to a common type and infers the stage count from `length(c)`.
"""
function Tableau(a::Matrix, b::Vector, e::Vector, c::Vector)
    T = promote_type(eltype.((a, b, e, c))...)
    Tableau{length(c)}(convert(Matrix{T}, a), convert.(Vector{T}, (b, e, c))...)
end

"""
    getindex(tab::Tableau, :a, i, j)
    getindex(tab::Tableau, :b | :e | :c, i)

Symbol-tagged accessor returning coefficients of the tableau by their
mathematical role: `:a` is the coupling matrix, `:b` and `:e` are the
main and embedded weights, and `:c` is the abscissae. The numeric
indices follow the same conventions as the underlying arrays.
"""
Base.getindex(tab::Tableau, ::Symbol, i::Integer, j::Integer) = tab.a[i, j]
function Base.getindex(tab::Tableau{T}, t::Symbol, i::Integer)::T where {T}
    t == :b && return tab.b[i]
    t == :e && return tab.e[i]
    t == :c && return tab.c[i]
end

"""
    convert(::Type{Tableau{T, NS}}, tab::Tableau{S, NS}) -> Tableau{T, NS}

Convert all coefficients of `tab` to element type `T`. The number of
stages must match.
"""
Base.convert(::Type{Tableau{T, NS}}, tab::Tableau{S, NS}) where {T, S, NS} =
    Tableau(convert(Matrix{T}, tab.a), convert(Vector{T}, tab.b),
            convert(Vector{T}, tab.e), convert(Vector{T}, tab.c))


# ============================================================================
# IMEXTableau — implicit + explicit pair
# ============================================================================

"""
    IMEXTableau{T, NS}(tabI::Tableau{T, NS}, tabE::Tableau{T, NS})

Bundle the implicit and explicit Butcher tableaux of an
implicit-explicit Runge–Kutta scheme. Both tableaux must share the
same element type and stage count.
"""
struct IMEXTableau{T, NS} <: AbstractTableau{T, NS}
    tabI::Tableau{T, NS}
    tabE::Tableau{T, NS}
end

"""
    IMEXTableau(tabI, tabE) -> IMEXTableau

Outer constructor that promotes the element types of `tabI` and
`tabE` to a common type before bundling them.
"""
function IMEXTableau(tabI::Tableau{TI, NS},
                     tabE::Tableau{TE, NS}) where {TI, TE, NS}
    T = promote_type(TI, TE)
    IMEXTableau(convert(Tableau{T, NS}, tabI), convert(Tableau{T, NS}, tabE))
end

"""
    convert(::Type{IMEXTableau{T, NS}}, tab) -> IMEXTableau{T, NS}

Convert both internal tableaux to element type `T`. The stage count
must match.
"""
Base.convert(::Type{IMEXTableau{T, NS}},
          tab::IMEXTableau{S, NS}) where {T, S, NS} =
    IMEXTableau(convert(Tableau{T, NS}, tab.tabI),
                convert(Tableau{T, NS}, tab.tabE))

"""
    getindex(tab::IMEXTableau, :aᴵ | :aᴱ, i, j)
    getindex(tab::IMEXTableau, :bᴵ | :eᴵ | :cᴵ | :bᴱ | :eᴱ | :cᴱ, i)

Coefficient accessor using superscripted symbols to disambiguate the
implicit (ᴵ) and explicit (ᴱ) parts of an IMEX tableau. Throws an
`ArgumentError` for unrecognised symbols.
"""
function Base.getindex(tab::IMEXTableau{T},
                         t::Symbol,
                         i::Integer,
                         j::Integer)::T where {T}
    t == :aᴵ && return tab.tabI[:a, i, j]
    t == :aᴱ && return tab.tabE[:a, i, j]
    throw(ArgumentError("symbol $t not recognized"))
end

function Base.getindex(tab::IMEXTableau{T}, t::Symbol, i::Integer)::T where {T}
    t == :bᴵ && return tab.tabI[:b, i]
    t == :eᴵ && return tab.tabI[:e, i]
    t == :cᴵ && return tab.tabI[:c, i]
    t == :bᴱ && return tab.tabE[:b, i]
    t == :eᴱ && return tab.tabE[:e, i]
    t == :cᴱ && return tab.tabE[:c, i]
    throw(ArgumentError("symbol $t not recognized"))
end


# ============================================================================
# Tableaux from Cavaglieri & Bewley 2015
# ============================================================================
#
# The schemes implemented in `src/steps/CB3R2R.jl` and `src/steps/CB4R3R.jl`
# rely on these named tableaux. Coefficients are entered as exact rationals
# and converted to `Float64` at the bottom of the file so the step kernels
# see a uniform element type.
#
# Cavaglieri, D. and Bewley, T., 2015. Low-storage implicit/explicit
# Runge–Kutta schemes for the simulation of stiff high-dimensional ODE
# systems. Journal of Computational Physics, 286, pp. 172–193.

# ~ IMEXRKCB2
const CB2_I = Tableau([0//1  0//1  0//1;
                       0//1  2//5  0//1;
                       0//1  5//6  1//6],
                      [0//1, 5//6, 1//6],
                      [0//1, 4//5, 1//5],
                      [0//1, 2//5, 1//1])

const CB2_E = Tableau([0//1  0//1  0//1;
                       2//5  0//1  0//1;
                       0//1  1//1  0//1],
                      [0//1, 5//6, 1//6],
                      [0//1, 4//5, 1//5],
                      [0//1, 2//5, 1//1])

# ~ CB3e
const CB3e_I = Tableau([0//1  0//1   0//1  0//1;
                        0//1  1//3   0//1  0//1;
                        0//1  1//2   1//2  0//1;
                        0//1  3//4  -1//4  1//2],
                       [0//1, 3//4, -1//4, 1//2],
                       [0//1, 3//4, -1//4, 1//2],
                       [0//1, 1//3,  1//1, 1//1])

const CB3e_E = Tableau([0//1  0//1   0//1  0//1;
                        1//3  0//1   0//1  0//1;
                        0//1  1//1   0//1  0//1;
                        0//1  3//4   1//4  0//1],
                       [0//1, 3//4, -1//4, 1//2],
                       [0//1, 3//4, -1//4, 1//2],
                       [0//1, 1//3,  1//1, 1//1])

# ~ CB3c
const CB3c_I = Tableau([0//1   0//1                                              0//1                         0//1;
                        0//1   3375509829940//4525919076317                      0//1                         0//1;
                        0//1  -11712383888607531889907//32694570495602105556248  566138307881//912153721139   0//1;
                        0//1   673488652607//2334033219546                       493801219040//853653026979   184814777513//1389668723319],
                       [0//1,  673488652607//2334033219546,                      493801219040//853653026979,  184814777513//1389668723319],
                       [0//1,  366319659506//1093160237145,                      270096253287//480244073137,  104228367309//1017021570740],
                       [0//1,  3375509829940//4525919076317,                     272778623835//1039454778728, 1//1])

const CB3c_E = Tableau([0//1                          0//1                          0//1                           0//1;
                        3375509829940//4525919076317  0//1                          0//1                           0//1;
                        0//1                          272778623835//1039454778728   0//1                           0//1;
                        0//1                          673488652607//2334033219546   1660544566939//2334033219546   0//1],
                       [0//1,                         673488652607//2334033219546,  493801219040//853653026979,    184814777513//1389668723319],
                       [449556814708//1155810555193,  0//1,                         210901428686//1400818478499,   480175564215//1042748212601],
                       [0//1,                         3375509829940//4525919076317, 272778623835//1039454778728,   1//1])


# ~ CB4
const CB4_I = Tableau([0//1                          0//1                          0//1                          0//1                         0//1                        0//1;
                       1//8                          1//8                          0//1                          0//1                         0//1                        0//1;
                       216145252607//961230882893    257479850128//1143310606989   30481561667//101628412017     0//1                         0//1                        0//1;
                       232049084587//1377130630063  -381180097479//1276440792700  -54660926949//461115766612     344309628413//552073727558   0//1                        0//1;
                       232049084587//1377130630063   322009889509//2243393849156  -100836174740//861952129159   -250423827953//1283875864443  1//2                        0//1;
                       232049084587//1377130630063   322009889509//2243393849156  -195109672787//1233165545817  -340582416761//705418832319   463396075661//409972144477  323177943294//1626646580633],
                      [232049084587//1377130630063,  322009889509//2243393849156, -195109672787//1233165545817, -340582416761//705418832319,  463396075661//409972144477, 323177943294//1626646580633],
                      [5590918588//49191225249,      92380217342//122399335103,   -29257529014//55608238079,    -126677396901//66917692409,   384446411890//169364936833, 58325237543//207682037557],
                      [0,                            1//4,                         3//4,                         3//8,                        1//2,                       1//1])

const CB4_E = Tableau([0//1                          0//1                          0//1                          0//1                          0//1                         0//1;
                       1//4                          0//1                          0//1                          0//1                          0//1                         0//1;
                       153985248130//1004999853329   902825336800//1512825644809   0//1                          0//1                          0//1                         0//1;
                       232049084587//1377130630063   99316866929//820744730663     82888780751//969573940619     0//1                          0//1                         0//1;
                       232049084587//1377130630063   322009889509//2243393849156   57501241309//765040883867     76345938311//676824576433     0//1                         0//1;
                       232049084587//1377130630063   322009889509//2243393849156  -195109672787//1233165545817  -4099309936455//6310162971841  1395992540491//933264948679  0//1],
                      [232049084587//1377130630063,  322009889509//2243393849156, -195109672787//1233165545817, -340582416761//705418832319,   463396075661//409972144477,  323177943294//1626646580633],
                      [5590918588//49191225249,      92380217342//122399335103,   -29257529014//55608238079,    -126677396901//66917692409,    384446411890//169364936833,  58325237543//207682037557],
                      [0,                            1//4,                         3//4,                         3//8,                         1//2,                        1//1])

# PR to add more are welcome! Defaults to Float64.
# FIXME: revisit when an adaptive variant with embedded error estimation is added.
const CB2  = convert(IMEXTableau{Float64, 3}, IMEXTableau(CB2_I,   CB2_E))
const CB3c = convert(IMEXTableau{Float64, 4}, IMEXTableau(CB3c_I, CB3c_E))
const CB3e = convert(IMEXTableau{Float64, 4}, IMEXTableau(CB3e_I, CB3e_E))
const CB4  = convert(IMEXTableau{Float64, 6}, IMEXTableau(CB4_I,   CB4_E))
