import Printf: @sprintf, Format, format

abstract type AbstractLogger end

# TODO: add fallback methods for some common subtypes (norm for arrays, etc.)
# TODO: change default false behaviour to just do nothing instead of printing to devnull
mutable struct Logger{N, O<:IO, F} <: AbstractLogger
          io::O   # where to print
     fstring::F   # Printf.Format string which gets printed
    logevery::Int # show every ... monitor calls

    Logger{N}(io::O, fstring::F, logevery) where {N, O, F} = new{N, O, F}(io, fstring, logevery)
end
Logger(io, monitor_out::NTuple{N, Any}, logevery) where {N} = Logger{N}(io, _generate_fstring_row(monitor_out), logevery)
Logger(io, monitor_out::Any, logevery) = Logger(io, (monitor_out,), logevery)
Logger(monitor_out, logevery) = Logger(stdout, monitor_out, logevery)

# change how much the logger prints output
set_logevery!(l::Logger, logevery) = (l.logevery = logevery; return l)

# generate the format of the string that gets printed each time the monitor is pushed to
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

# print the header given the number of tracked values in the monitor
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

# printf row of monitored values
function (l::Logger{N})(count, store::AbstractStorage) where {N}
    if count % l.logevery == 0
        s = format(l.fstring, times(store)[end], _filter_outputs(samples(store)[end])...)
        println(l.io, s)
        flush(l.io)
    end
end

# TODO: this function needs some damn improvement
function _filter_outputs(out)
    l = []

    # if out if just an iterable array then don't do anything
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
