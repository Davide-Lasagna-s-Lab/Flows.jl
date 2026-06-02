import Printf: @sprintf, Format, format

"""
    AbstractLogger

Supertype for objects that format and emit the state of a
[`Monitor`](@ref) to an output stream. Subtypes must be callable in two
forms: `l()` to print the header, and `l(count, store)` to print one
sample row from `store`.
"""
abstract type AbstractLogger end

# TODO: add fallback methods for some common subtypes (norm for arrays, etc.)
# TODO: change default-`false` behaviour to do nothing instead of printing to devnull

"""
    Logger{N, O<:IO, F}(io, fstring, logevery) <: AbstractLogger

Concrete logger that prints the state of a [`Monitor`](@ref) to `io`
using a precompiled `Printf.Format`. `N` is the number of observed
quantities per sample (which controls the width of the printed row).
`logevery` decimates the output: only every `logevery`-th sample is
emitted, but the header is always printed once before the first row.

Construct via the [`Logger`](@ref) outer constructors below.
"""
mutable struct Logger{N, O<:IO, F} <: AbstractLogger
          io::O   # output stream
     fstring::F   # precompiled Printf format used for each row
    logevery::Int # print one row every `logevery` monitor calls

    Logger{N}(io::O, fstring::F, logevery) where {N, O, F} = new{N, O, F}(io, fstring, logevery)
end

"""
    Logger(io, monitor_out, logevery) -> Logger
    Logger(monitor_out, logevery)     -> Logger

Construct a [`Logger`](@ref). `monitor_out` is a sample produced by the
monitor's observable (used only to infer the printable type of each
column); when it is a tuple, one column per element is generated.
The single-argument convenience overload defaults `io` to `stdout`.
"""
Logger(io, monitor_out::NTuple{N, Any}, logevery) where {N} =
    Logger{N}(io, _generate_fstring_row(monitor_out), logevery)
Logger(io, monitor_out::Any, logevery) = Logger(io, (monitor_out,), logevery)
Logger(monitor_out, logevery) = Logger(stdout, monitor_out, logevery)

"""
    set_logevery!(l::Logger, logevery::Int) -> l

Change how often `l` prints. Returns `l` to allow chaining.
"""
set_logevery!(l::Logger, logevery) = (l.logevery = logevery; return l)

# Build a `Printf.Format` covering the time column plus one column per
# observed quantity. Format choice per column depends on the column type
# at logger construction time: floats, integers and complex numbers each
# get their own width/precision; anything else is rendered as `?`.
function _generate_fstring_row(monitor_out::NTuple{N, Any}) where {N}
    s = "| %5.3e "
    for i in 1:N
        s = s*_generate_fstring(monitor_out[i])
    end
    s = s*"|"
    return Format(s)
end

_generate_fstring(::AbstractFloat) = "|  % 5.5e  "
_generate_fstring(::Integer)       = "|     %6i     "
_generate_fstring(out::Complex)    = "| % 5.4f%+5.4fi"
_generate_fstring(::Any)           = "|        ?       "

# Print the table header. Each call prints three lines: top border,
# titles, and a second border.
function (l::Logger{N})() where {N}
    # print header upper border
    print(l.io, "+-----------")
    @inbounds for i in 1:N
        print(l.io, "+----------------")
    end
    println(l.io, "+")

    # print header titles
    print(l.io, "|     t     ")
    @inbounds for i in 1:N
        print(l.io, "|   Monitor #$i   ")
    end
    println(l.io, "|")

    # print header upper border
    print(l.io, "+-----------")
    @inbounds for i in 1:N
        print(l.io, "+----------------")
    end
    println(l.io, "+")

    # flush output stream to print
    flush(l.io)
end

# Print one row of monitored values, decimated by `logevery`.
function (l::Logger{N})(count, store::AbstractStorage) where {N}
    if count % l.logevery == 0
        s = format(l.fstring, times(store)[end], _filter_outputs(samples(store)[end])...)
        println(l.io, s)
        flush(l.io)
    end
end

# TODO: this function needs some improvement; it does not handle nested
# containers consistently.
#
# Flatten a single monitor sample into a list of scalar values that the
# format string can consume. `Complex` is split into its real and
# imaginary parts; arrays are skipped (rendered as `?` columns); other
# numeric types are forwarded verbatim.
function _filter_outputs(out)
    l = []

    # if out is an iterable array then don't do anything
    if out isa AbstractArray
        return l
    end

    # otherwise iterate over monitor output and assign to filtered list
    for o in out
        if o isa Complex
            push!(l, real(o), imag(o))
        elseif o isa Number
            push!(l, o)
        end
    end
    return l
end
