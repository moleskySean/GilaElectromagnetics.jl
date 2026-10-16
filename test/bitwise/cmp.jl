# Compare two snapshots of snap.jl: one line per output that differs, with how many entries differ, the max-norm difference over
# the max-norm and the largest per-entry relative difference (entries above 1e-8 of the max), in eps of the element type. Exits 1 on any difference.
using DoubleFloats, Printf, Serialization

function dif(X::AbstractArray, Y::AbstractArray)
    size(X) == size(Y) || return "size $(size(X)) vs $(size(Y))"
    e = eps(real(eltype(Y))); m = maximum(abs, Y)
    pe = maximum((abs(X[i] - Y[i]) / abs(Y[i]) for i in eachindex(Y) if abs(Y[i]) > 1e-8 * m); init = 0)
    @sprintf("%d of %d entries, %.3g eps max-norm, %.3g eps per entry", count(.!isequal.(X, Y)), length(Y), maximum(abs, X .- Y) / m / e, pe / e)
end
dif(X::Tuple, Y::Tuple) = join(map(dif, X, Y), "; ")
dif(X::AbstractVector{<:AbstractArray}, Y::AbstractVector{<:AbstractArray}) = join(map(dif, X, Y), "; ")
dif(X, Y) = "$(repr(X)) vs $(repr(Y))"

A, B = deserialize(ARGS[1]), deserialize(ARGS[2])
bad = 0
for key in sort(collect(union(keys(A), keys(B))))
    haskey(A, key) && haskey(B, key) || (println(key, ": only in ", haskey(A, key) ? ARGS[1] : ARGS[2]); global bad += 1; continue)
    isequal(A[key], B[key]) && continue
    println(key, ": ", dif(A[key], B[key])); global bad += 1
end
println(length(union(keys(A), keys(B))) - bad, " bitwise, ", bad, " different")
exit(bad == 0 ? 0 : 1)
