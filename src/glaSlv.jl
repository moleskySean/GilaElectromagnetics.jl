using LinearAlgebra
using Adapt
using KernelAbstractions: get_backend

"""
    isvarying(preCon)

Whether `preCon` applies a different linear map on different applications.

The question is linearity in the vector handed over.

Defaults to `false`, so a preconditioner that varies has to say so.
"""
isvarying(preCon) = false
# ldiv! on a Gila operator is an inner Krylov solve run to a tolerance
isvarying(::AbstractGlaOpr) = true

#= A Krylov method builds its space from one fixed linear map. Under a varying
preconditioner the Arnoldi relation quietly stops holding, and the residual the
solver reports stops being the residual of anything. =#
chkVry(preCon, flx = false) = !flx && isvarying(preCon) &&
    throw(ArgumentError("$(typeof(preCon)) applies a different linear map on every application, and this solver assumes a fixed one: it would converge to a believable wrong answer rather than fail. Solve with `GMRESSolver(; flexible = true)`, precondition with a fixed linear map (a factorization, a diagonal, a fixed number of stationary sweeps), or define `isvarying(::$(typeof(preCon))) = false` if the map really is fixed."))

"""
    SlvLog(; truSmp = :atrestart, hssSmp = :all)

A record of one `solve`, filled in place when passed as `solve(opr, inp, slv; log = SlvLog())`.

# Arguments
- `truSmp::Symbol`: `:atrestart` (default) samples `resTru` at every GMRES restart, `:never` only at exit
- `hssSmp::Symbol`: `:all` (default) keeps every Hessenberg cycle, `:first` only the first

# Fields
- `status::Symbol`: `:converged`, `:maxiter`, `:breakdown`, or `:none` before a solve fills it
- `numItr::Int`: Iterations taken
- `resRec::Vector{Float64}`: The relative recursive residual the algorithm tracks, before the first step and after each one
- `resTru::Vector{Float64}`: The true relative residual `‖inp - opr * out‖ / ‖inp‖`, exit value last
- `hss::Vector{Matrix{ComplexF64}}`: The GMRES Hessenberg matrix per restart cycle, as built, augmented by the recycled directions for a `GCRODRSolver`
- `oprApp::Int`: Operator applications made by the solver
- `pcnApp::Int`: Preconditioner applications made by the solver
- `oprAppLog::Int`: Operator applications made to fill this record, kept out of `oprApp`
- `oprAppPcn::Int`: Operator applications made inside `ldiv!`, which only the preconditioner can count: left at zero here
- `ρ`, `α`, `ω`, `dnm::Vector{ComplexF64}`: BiCGStab's scalars and its `dot(resShd, v)` denominator, per iteration
- `rcyErr::Vector{Float64}`: `‖CᴴC - I‖` for a `GCRODRSolver`'s recycled subspace, per cycle
- `prm::NamedTuple`: The settings the solve ran under, resolved and rescaled, and the iterations restarts fell on
- `inner::Vector{SlvLog}`: One record per inner solve, for a solver that wraps another
"""
mutable struct SlvLog
    status::Symbol
    numItr::Int
    resRec::Vector{Float64}
    resTru::Vector{Float64}
    hss::Vector{Matrix{ComplexF64}}
    oprApp::Int
    pcnApp::Int
    oprAppLog::Int
    oprAppPcn::Int
    ρ::Vector{ComplexF64}
    α::Vector{ComplexF64}
    ω::Vector{ComplexF64}
    dnm::Vector{ComplexF64}
    rcyErr::Vector{Float64}
    prm::NamedTuple
    inner::Vector{SlvLog}
    truSmp::Symbol
    hssSmp::Symbol
    function SlvLog(; truSmp = :atrestart, hssSmp = :all)
        truSmp in (:never, :atrestart) ||
            throw(ArgumentError("The true residual is sampled :never or :atrestart, not :$truSmp. Sampling it every iteration costs a matvec per step and is not offered."))
        hssSmp in (:all, :first) ||
            throw(ArgumentError("Hessenberg cycles are kept :all or :first, not :$hssSmp."))
        return new(:none, 0, Float64[], Float64[], Matrix{ComplexF64}[], 0, 0, 0, 0,
            ComplexF64[], ComplexF64[], ComplexF64[], ComplexF64[], Float64[],
            NamedTuple(), SlvLog[], truSmp, hssSmp)
    end
end

# A zero right hand side offers no scale, so residuals against it stay absolute
nrmRel(inp) = (nrm = norm(inp); iszero(nrm) ? one(nrm) : nrm)

#= Fill the exit record. The true residual is worth its one matvec because
neither solver forms it while iterating: GMRES stops on a preconditioned
quantity, BiCGStab on a recursive one that is never refreshed. =#
function logEnd!(log, opr, inp, out, status, numItr)
    isnothing(log) && return out
    log.status = status
    log.numItr = numItr
    act = similar(vec(out))
    mul!(act, opr, vec(out))
    log.oprAppLog += 1
    push!(log.resTru, norm(vec(inp) .- act) / nrmRel(inp))
    return out
end

"""
    GMRESSolver

An iterative solver for linear systems of equations that uses the Generalized Minimal
Residual Method (GMRES). This method is particularly effective for non-symmetric
linear systems.

# Fields
- `rstItr::Union{Nothing, Int}`: Number of iterations until restart (default: min(20, length(vec)))
- `maxItr::Union{Nothing, Int}`: Maximum number of iterations (default: max(5000, length(vec)))
- `absTol::Union{Nothing, Real}`: Absolute tolerance for convergence (default: 0)
- `relTol::Union{Nothing, Real}`: Relative tolerance for convergence (default: √ε)
- `preCon`: Preconditioner, anything with an `ldiv!(out, preCon, inp)` method (default: none)
- `side::Symbol`: `:left` (default) or `:right`, the side `preCon` is applied on
- `flexible::Bool`: Whether to store the preconditioned basis, admitting a varying `preCon` (default: false)

On the left the Krylov space is that of `preCon⁻¹ * opr` and the quantity
driving the stopping test is `preCon⁻¹ * (inp - opr * out)`, so two
preconditioners reporting the same relative residual have not converged to the
same thing: the discrepancy is the size of `cond(preCon)`. On the right the
space is that of `opr * preCon⁻¹`, the residual is `inp - opr * out` itself, and
the price is one extra `ldiv!` and one extra pass over the solution per restart
cycle.

Flexible GMRES keeps `Z = [z₁ … z_k]` with `zₖ = preConₖ⁻¹ * vₖ` and updates
`out += Z * y`, so the applications never have to agree with each other. It
implies `side = :right`, and it costs `rstItr` extra full length vectors.
"""
struct GMRESSolver <: GlaSlv
    rstItr::Union{Nothing, Int} # Iterations until restart
    maxItr::Union{Nothing, Int} # Maximum number of iterations
    absTol::Union{Nothing, Real} # Absolute tolerance
    relTol::Union{Nothing, Real} # Relative tolerance
    preCon::Any # Preconditioner
    side::Symbol # Side the preconditioner is applied on
    flexible::Bool # Whether the preconditioned basis is stored
end

"""
    GMRESSolver(rstItr = nothing, maxItr = nothing, absTol = nothing, relTol = nothing; preCon = nothing, flexible = false, side = flexible ? :right : :left)

Create a GMRESSolver. Unset values are resolved against the right hand side at
solve time.

# Returns
- `GMRESSolver`: A new solver instance
"""
function GMRESSolver(rstItr = nothing, maxItr = nothing, absTol = nothing,
    relTol = nothing; preCon = nothing, flexible = false,
    side = flexible ? :right : :left)
    side in (:left, :right) ||
        throw(ArgumentError("A preconditioner is applied on the :left or on the :right, not on the :$side."))
    flexible && side == :left &&
        throw(ArgumentError("Flexible GMRES stores the basis the preconditioner produces, which only the :right space has: there is no :left variant of it."))
    return GMRESSolver(rstItr, maxItr, absTol, relTol, preCon, side, flexible)
end

# Solver settings with the unset ones filled in from the right hand side
slvPrm(slv::GMRESSolver, vec::AbstractArray) = (
    rstItr = @something(slv.rstItr, min(20, length(vec))),
    maxItr = @something(slv.maxItr, max(5000, length(vec))),
    absTol = @something(slv.absTol, zero(real(eltype(vec)))),
    relTol = @something(slv.relTol, sqrt(eps(real(eltype(vec))))))

"""
    BiCGStabSolver

An iterative solver for linear systems of equations that uses the BiConjugate
Gradient Stabilized Method (BiCGStab). This method is effective for non-symmetric
linear systems and typically requires less memory than GMRES.

# Fields
- `maxItr::Union{Nothing, Int}`: Maximum number of iterations (default: length(vec))
- `absTol::Union{Nothing, Real}`: Absolute tolerance for convergence (default: 0)
- `relTol::Union{Nothing, Real}`: Relative tolerance for convergence (default: √ε)
- `preCon`: Right preconditioner, anything with an `ldiv!(out, preCon, inp)` method (default: none)

The preconditioner is applied on the right: the Krylov space is that of
`opr * preCon⁻¹`, and the residual driving the stopping test is `inp - opr * out`
itself, not a preconditioned version of it.
"""
struct BiCGStabSolver <: GlaSlv
    maxItr::Union{Nothing, Int} # Maximum number of iterations
    absTol::Union{Nothing, Real} # Absolute tolerance
    relTol::Union{Nothing, Real} # Relative tolerance
    preCon::Any # Right preconditioner
end

"""
    BiCGStabSolver(maxItr = nothing, absTol = nothing, relTol = nothing; preCon = nothing)

Create a BiCGStabSolver. Unset values are resolved against the right hand side
at solve time.

# Returns
- `BiCGStabSolver`: A new solver instance
"""
BiCGStabSolver(maxItr = nothing, absTol = nothing, relTol = nothing; preCon = nothing) =
    BiCGStabSolver(maxItr, absTol, relTol, preCon)

# Solver settings with the unset ones filled in from the right hand side
slvPrm(slv::BiCGStabSolver, vec::AbstractVector) = (
    maxItr = @something(slv.maxItr, length(vec)),
    absTol = @something(slv.absTol, zero(real(eltype(vec)))),
    relTol = @something(slv.relTol, sqrt(eps(real(eltype(vec))))))

"""
    solve(opr, inp::AbstractVector{T}, slv::BiCGStabSolver; x0 = nothing, log = nothing) where T

Solve the linear system `opr * out = inp` using the BiCGStab method.

# Arguments
- `opr`: The operator in the linear system, anything supporting `mul!` and `size`
- `inp::AbstractVector{T}`: The right-hand side vector
- `slv::BiCGStabSolver`: The solver parameters
- `x0`: An initial guess, or `nothing` (default) to start from zero
- `log`: A `SlvLog` to fill in place, or `nothing` (default) for no record

# Returns
- `out::AbstractVector{T}`: The solution vector

# Notes
- The solver will iterate until either convergence is achieved or the maximum
  number of iterations is reached
- Convergence is determined by the absolute and relative tolerances specified
  in the solver parameters
"""
function solve(opr, inp::AbstractVector{T}, slv::BiCGStabSolver; x0 = nothing,
    log = nothing) where T
    (; maxItr, absTol, relTol) = slvPrm(slv, inp)
    preCon = slv.preCon
    chkVry(preCon)
    out = isnothing(x0) ? fill!(similar(inp), zero(T)) : copyto!(similar(inp), x0)

    ρPrv = zero(T)
    ω = zero(T)
    α = zero(T)
    # Work buffers, allocated once (zeroed so a β = 0 `mul!` never reads garbage)
    v = fill!(similar(inp), zero(T))
    t = fill!(similar(inp), zero(T))
    res = similar(inp) # Residual
    if isnothing(x0)
        copyto!(res, inp)
    else
        mul!(res, opr, out)
        isnothing(log) || (log.oprApp += 1)
        res .= inp .- res
    end
    #= relTol is measured against the right hand side rather than against the
    initial residual: a good guess shrinks ‖res₀‖, and scaling by it would hold
    a warm start to a tighter absolute threshold than a cold one. =#
    absTol = max(absTol, relTol * norm(inp))
    nrmRhs = nrmRel(inp)
    isnothing(log) || (log.prm = (maxItr = maxItr, absTol = absTol, relTol = relTol,
        side = :right, flexible = false, elmTyp = T,
        arrTyp = Base.typename(typeof(inp)).wrapper))

    resShd = copy(res) # Residual shadow
    p = copy(res)
    s = similar(res)
    # ldiv! must not clobber p or s, both of which are read again below
    p̂ = isnothing(preCon) ? p : similar(p)
    ŝ = isnothing(preCon) ? s : similar(s)

    brkTol = eps(real(T))
    nrmShd = norm(resShd)

    for numItr in 1:maxItr
        nrmRes = norm(res)
        isnothing(log) || push!(log.resRec, nrmRes / nrmRhs)
        if nrmRes < absTol
            return logEnd!(log, opr, inp, out, :converged, numItr - 1)
        end
        isfinite(nrmRes) || return logEnd!(log, opr, inp, out, :breakdown, numItr - 1)

        ρ = dot(resShd, res)
        isnothing(log) || push!(log.ρ, ρ)
        if numItr > 1
            β = (ρ / ρPrv) * (α / ω)
            p .= res .+ β .* (p .- ω .* v)
        end
        isnothing(preCon) || ldiv!(p̂, preCon, p)
        isnothing(log) || isnothing(preCon) || (log.pcnApp += 1)
        mul!(v, opr, p̂)
        isnothing(log) || (log.oprApp += 1)
        dnm = dot(resShd, v)
        isnothing(log) || push!(log.dnm, dnm)
        abs(dnm) <= brkTol * nrmShd * norm(v) &&
            return logEnd!(log, opr, inp, out, :breakdown, numItr)
        α = ρ / dnm
        isnothing(log) || push!(log.α, α)
        res .-= α .* v
        s .= res

        nrmRes = norm(res)
        isnothing(log) || push!(log.resRec, nrmRes / nrmRhs)
        if nrmRes < absTol
            out .+= α .* p̂
            return logEnd!(log, opr, inp, out, :converged, numItr)
        end

        isnothing(preCon) || ldiv!(ŝ, preCon, s)
        isnothing(log) || isnothing(preCon) || (log.pcnApp += 1)
        mul!(t, opr, ŝ)
        isnothing(log) || (log.oprApp += 1)
        dnm = dot(t, t)
        # ω's numerator is dot(t, s), and s is res, so nrmRes sets the scale here
        abs(dnm) <= brkTol * norm(t) * nrmRes &&
            return logEnd!(log, opr, inp, out, :breakdown, numItr)
        ω = dot(t, s) / dnm
        isnothing(log) || push!(log.ω, ω)
        # ω = dot(t, res) / dot(t, t)
        out .+= α .* p̂ .+ ω .* ŝ
        res .-= ω .* t
        ρPrv = ρ
    end
    @warn "BiCGStab did not converge after $maxItr iterations."
    return logEnd!(log, opr, inp, out, :maxiter, maxItr)
end

"""
    solve(opr, inp::AbstractArray{T}, slv::GMRESSolver; x0 = nothing, log = nothing) where T

Solve the linear system `opr * out = inp` using the GMRES method.

# Arguments
- `opr`: The operator in the linear system, anything supporting `mul!` and `size`
- `inp::AbstractArray{T}`: The right-hand side vector
- `slv::GMRESSolver`: The solver parameters
- `x0`: An initial guess, or `nothing` (default) to start from zero
- `log`: A `SlvLog` to fill in place, or `nothing` (default) for no record

# Returns
- `out::AbstractArray{T}`: The solution vector

# Notes
- The solver will iterate until either convergence is achieved or the maximum
  number of iterations is reached
- Convergence is determined by the absolute and relative tolerances specified
  in the solver parameters
- The solver uses restarts to manage memory usage, with the restart frequency
  specified in the solver parameters
"""
function solve(opr, inp::AbstractArray{T}, slv::GMRESSolver; x0 = nothing, log = nothing) where T
    # Algorithm adapted from https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/0b2f1c5d352069df1bc891750087deda2d14cc9d/src/gmres.jl

    (; rstItr, maxItr, absTol, relTol) = slvPrm(slv, inp)
    preCon = slv.preCon
    chkVry(preCon, slv.flexible)
    # Every variant agrees when there is nothing to apply
    sid = isnothing(preCon) ? :none : (slv.flexible ? :flex : slv.side)

    basVec = similar(inp, size(opr, 1), 1 + rstItr) # Krylov basis vectors
    hss = zeros(eltype(inp), 1 + rstItr, rstItr) # Hessenberg matrix (always stored on the CPU)
    nllSpc = ones(eltype(inp), 1 + rstItr) # Vector in the nullspace of the Hessenberg matrix (always stored on the CPU)
    out = isnothing(x0) ? fill!(similar(inp), zero(T)) : copyto!(similar(inp), x0) # Solution vector
    buf = fill!(similar(out), zero(T)) # Work buffer for the restart residual
    # ldiv! is out of place, so what goes into the preconditioner needs its own buffer
    pcnBuf = isnothing(preCon) || sid == :flex ? nothing : similar(inp, size(opr, 1))
    #= zₖ = preConₖ⁻¹ * vₖ, kept so the update can be taken against it. Its own
    column is the ldiv! destination, and the update only ever reads 1:k-1. =#
    pcdVec = sid == :flex ? similar(inp, size(opr, 1), rstItr) : nothing

    # The first basis vector is the initial residual (preconditioned on the left)
    k = 1 # Which restart iteration we are on
    vk = @view basVec[:, k]
    if isnothing(x0)
        sid == :left ? ldiv!(vk, preCon, inp) : copyto!(vk, inp) # b - opr * x with x = 0
    else
        mul!(buf, opr, out)
        isnothing(log) || (log.oprApp += 1)
        if sid == :left
            pcnBuf .= vec(inp) .- vec(buf)
            ldiv!(vk, preCon, pcnBuf)
        else
            vk .= vec(inp) .- vec(buf)
        end
    end
    isnothing(log) || sid == :left && (log.pcnApp += 1)
    β = norm(vk)
    iszero(β) || rmul!(vk, inv(β)) # An exact guess leaves no direction to normalize

    # Save the residual
    res = β # Residual
    resAcc = one(real(T)) # Residual accumulator

    #= relTol is measured against the right hand side rather than against the
    initial residual: a good guess shrinks ‖res₀‖, and scaling by it would hold
    a warm start to a tighter absolute threshold than a cold one. The left
    preconditioned residual lives in preCon⁻¹'s image, so the anchor does too. =#
    if isnothing(x0)
        nrmAnc = β
    elseif sid == :left
        ldiv!(pcnBuf, preCon, inp)
        isnothing(log) || (log.pcnApp += 1)
        nrmAnc = norm(pcnBuf)
    else
        nrmAnc = norm(inp)
    end
    absTol = max(absTol, relTol * nrmAnc)
    iszero(nrmAnc) && (nrmAnc = one(nrmAnc))

    # resRec is relative to what the stopping test is anchored to, resTru to ‖inp‖
    nrmRhs = isnothing(log) ? NaN : Float64(nrmRel(inp))
    rstIdx = isnothing(log) ? nothing : Int[] # Iterations a restart fell on
    isnothing(log) || push!(log.resRec, res / nrmAnc)

    numItr = 0 # Arnoldi steps taken
    for itr in 1:maxItr
        if res <= absTol || !isfinite(res)
            break
        end
        numItr = itr

        # vₖ₊₁ = precon \ (opr * vₖ), or opr * (precon \ vₖ) on the right
        vkp1 = @view basVec[:, k + 1]
        if sid == :left
            mul!(pcnBuf, opr, vk)
            ldiv!(vkp1, preCon, pcnBuf)
        elseif sid == :none
            mul!(vkp1, opr, vk)
        else
            z = sid == :flex ? view(pcdVec, :, k) : pcnBuf
            ldiv!(z, preCon, vk)
            mul!(vkp1, opr, z)
        end
        isnothing(log) || (log.oprApp += 1)
        isnothing(log) || sid == :none || (log.pcnApp += 1)

        # Orthogonalize vₖ₊₁ against previous basis vectors
        for j in 1:k
            col = @view basVec[:, j]
            hss[j, k] = dot(col, vkp1)
            axpy!(-hss[j, k], col, vkp1) # vkp1 .-= hessenberg[j, k] .* col
        end

        # Normalize vₖ₊₁ and compute the residual
        hss[k + 1, k] = norm(vkp1)
        #= A vanishing subdiagonal means span(v₁ … vₖ) is already invariant
        under opr, so the least squares solution over it is exact. vₖ₊₁ is then
        not a direction at all: it is left as the zero it is rather than
        normalized into a column of NaNs. =#
        if iszero(hss[k + 1, k])
            res = zero(res)
        else
            rmul!(vkp1, inv(hss[k + 1, k]))
            nllSpc[k + 1] = -conj(dot(view(nllSpc, 1:k), view(hss, 1:k, k)) / hss[k + 1, k]) # update the nullspace vector
            resAcc += real(abs2(nllSpc[k+1]))
            res = β / sqrt(resAcc)
        end
        isnothing(log) || push!(log.resRec, res / nrmAnc)

        # Next iteration
        k += 1
        vk = vkp1

        # At the end of restarts (and/or when we have converged/maxed out the iterations), update our solution vector
        don = (res <= absTol || itr == maxItr) # done or not
        if don || k == 1 + rstItr
            # Solve the least squares problem: Hy = βe₁
            y = fill!(similar(hss, (k,)), zero(T))
            y[1] = β
            #= Ritz values, harmonic Ritz values and condition estimates all
            come out of the raw Hessenberg and none of them come back, so it is
            the matrix that is kept, taken before lstSqrHss overwrites it. =#
            if !isnothing(log) && (log.hssSmp == :all || isempty(log.hss))
                push!(log.hss, Matrix{ComplexF64}(view(hss, 1:k, 1:k-1)))
            end
            lstSqrHss(view(hss, 1:k, 1:k-1), y) # Note: this mutates `hss`, but it's fine because we don't need it anymore
            y = adapt(get_backend(out), y) # Copy to the device if needed

            # Update the solution vector
            if sid == :right
                #= The space on the right is that of opr * precon⁻¹, so what
                the least squares problem returns is precon applied to the
                solution update rather than the update itself. Dividing it back
                once per cycle costs one ldiv! per restart, not one per step. =#
                mul!(buf, view(basVec, :, 1:k-1), view(y, 1:k-1))
                ldiv!(pcnBuf, preCon, buf)
                isnothing(log) || (log.pcnApp += 1)
                vec(out) .+= pcnBuf
            else
                # Z y on the flexible path is already the update itself
                updVec = sid == :flex ? pcdVec : basVec
                mul!(out, view(updVec, :, 1:k-1), view(y, 1:k-1), one(T), one(T))
            end

            k = 1

            # If we have not reached max_iter or converged, restart
            if !don
                vk = @view basVec[:, k]
                mul!(buf, opr, out)
                isnothing(log) || (log.oprApp += 1)
                isnothing(log) || push!(rstIdx, itr)
                if sid == :left
                    pcnBuf .= vec(inp) .- vec(buf)
                    isnothing(log) || log.truSmp == :never ||
                        push!(log.resTru, norm(pcnBuf) / nrmRhs)
                    ldiv!(vk, preCon, pcnBuf)
                    isnothing(log) || (log.pcnApp += 1)
                else
                    vk .= vec(inp) .- vec(buf)
                    isnothing(log) || log.truSmp == :never ||
                        push!(log.resTru, norm(vk) / nrmRhs)
                end
                β = norm(vk)
                rmul!(vk, inv(β))
                resAcc = one(real(T))
            end
        end
    end
    isnothing(log) || (log.prm = (rstItr = rstItr, maxItr = maxItr, absTol = absTol,
        relTol = relTol, side = sid == :flex ? :right : slv.side,
        flexible = slv.flexible, elmTyp = T,
        arrTyp = Base.typename(typeof(inp)).wrapper, rstIdx = rstIdx))
    status = !isfinite(res) ? :breakdown : (res <= absTol ? :converged : :maxiter)
    status == :maxiter && @warn "GMRES failed to converge after $maxItr iterations"
    return logEnd!(log, opr, inp, out, status, numItr)
end

"""
    GCRODRSolver

An iterative solver for linear systems of equations that uses GCRO with deflated
restarting (GCRO-DR). GCRO-DR keeps `rcyDim` directions of the Krylov space
built in GMRES and reuses them in the following cycle and, through the `rcy`
argument of `solve`, in a subsequent solve.

# Fields
- `rstItr::Union{Nothing, Int}`: Iterations in a cycle, `rcyDim` included (default: min(20, length(vec)))
- `maxItr::Union{Nothing, Int}`: Maximum number of iterations (default: max(5000, length(vec)))
- `absTol::Union{Nothing, Real}`: Absolute tolerance for convergence (default: 0)
- `relTol::Union{Nothing, Real}`: Relative tolerance for convergence (default: √ε)
- `preCon`: Preconditioner, anything with an `ldiv!(out, preCon, inp)` method (default: none)
- `side::Symbol`: `:left` (default) or `:right`, the side `preCon` is applied on
- `rcyDim::Int`: Directions carried between cycles, `0` gives plain GMRES
- `lrgFrc::Float64`: Fraction of `rcyDim` taken from the large end of the spectrum

The recycled pair `(U, C)` satisfies `A * U = C` with `C' * C = I`, `A` being the
preconditioned operator. The best correction inside `range(U)` therefore costs no
operator applications, and the cycle's least squares problem decouples into the
plain GMRES one.

`lrgFrc * rcyDim` of the directions approximate the eigenvectors of largest
modulus (through ordinary Ritz values) and the rest those of smallest modulus
(through harmonic Ritz values). Deflating the small end (making `lrgFrc` closer
to `0`) changes the rate of
convergence.

There is no `flexible` field: flexible GMRES stores the preconditioned basis,
which does not satisfy `A * U = C`.
"""
struct GCRODRSolver <: GlaSlv
    rstItr::Union{Nothing, Int} # Iterations in a cycle, the recycled directions included
    maxItr::Union{Nothing, Int} # Maximum number of iterations
    absTol::Union{Nothing, Real} # Absolute tolerance
    relTol::Union{Nothing, Real} # Relative tolerance
    preCon::Any # Preconditioner
    side::Symbol # Side the preconditioner is applied on
    rcyDim::Int # Directions carried between cycles
    lrgFrc::Float64 # Fraction of rcyDim taken from the large end of the spectrum
end

"""
    GCRODRSolver(rstItr = nothing, maxItr = nothing, absTol = nothing, relTol = nothing; preCon = nothing, side = :left, rcyDim = 10, lrgFrc = 0.0)

Create a GCRODRSolver. Unset values are resolved against the right hand side at
solve time.

# Returns
- `GCRODRSolver`: A new solver instance
"""
function GCRODRSolver(rstItr = nothing, maxItr = nothing, absTol = nothing,
    relTol = nothing; preCon = nothing, side = :left, rcyDim = 10, lrgFrc = 0.0)
    side in (:left, :right) ||
        throw(ArgumentError("A preconditioner is applied on the :left or on the :right. Side :$side is not supported."))
    0 <= lrgFrc <= 1 ||
        throw(ArgumentError("lrgFrc ∈ [0, 1] is the fraction of the recycled subspace taken from the large end of the spectrum. Value of $lrgFrc is not supported."))
    rcyDim >= 0 ||
        throw(ArgumentError("We require the rcyDim >= 0. Given $rcyDim is not supported."))
    isnothing(rstItr) || rcyDim < rstItr ||
        throw(ArgumentError("A cycle of $rstItr iterations carrying $rcyDim recycled directions leaves no Arnoldi step to take."))
    return GCRODRSolver(rstItr, maxItr, absTol, relTol, preCon, side, rcyDim, lrgFrc)
end

# Solver settings with the unset ones filled in from the right hand side
function slvPrm(slv::GCRODRSolver, vec::AbstractArray)
    rstItr = @something(slv.rstItr, min(20, length(vec)))
    return (rstItr = rstItr,
        maxItr = @something(slv.maxItr, max(5000, length(vec))),
        absTol = @something(slv.absTol, zero(real(eltype(vec)))),
        relTol = @something(slv.relTol, sqrt(eps(real(eltype(vec))))),
        # A tiny right hand side must not leave a cycle with zero Arnoldi steps
        rcyDim = min(slv.rcyDim, rstItr - 1))
end

"""
    RcySpc()

The subspace a `GCRODRSolver` recycles, filled in place when passed as
`solve(opr, inp, slv; rcy = RcySpc())`, and read by the next solve it is passed to.

# Fields
- `bas`: `U`, the basis of the subspace, or `nothing` before a solve fills it
- `img`: `C`, equal to `A * U` with orthonormal columns, `A` being the preconditioned operator
"""
mutable struct RcySpc
    bas::Any # U
    img::Any # C = A * U
end

RcySpc() = RcySpc(nothing, nothing)

"""
    solve(opr, inp::AbstractArray{T}, slv::GCRODRSolver; x0 = nothing, log = nothing, rcy = nothing) where T

Solve the linear system `opr * out = inp` using the GCRO-DR method.

# Arguments
- `opr`: The operator in the linear system, anything supporting `mul!` and `size`
- `inp::AbstractArray{T}`: The right-hand side vector
- `slv::GCRODRSolver`: The solver parameters
- `x0`: An initial guess, or `nothing` (default) to start from zero
- `log`: A `SlvLog` to fill in place, or `nothing` (default) for no record
- `rcy`: A `RcySpc` to start from and fill in place, or `nothing` (default) to
  recycle within this solve only

# Returns
- `out::AbstractArray{T}`: The solution vector

# Notes
- A supplied subspace is checked against the operator of this solve before it is
  trusted, and rebuilt when it does not match
- `log.hss` holds the augmented matrix `[I B; 0 H]` of the cycle, which is
  Hessenberg only when `rcyDim` is zero
"""
function solve(opr, inp::AbstractArray{T}, slv::GCRODRSolver; x0 = nothing,
    log = nothing, rcy = nothing) where T
    (; rstItr, maxItr, absTol, relTol, rcyDim) = slvPrm(slv, inp)
    preCon = slv.preCon
    chkVry(preCon)
    # Every variant agrees when there is nothing to apply
    sid = isnothing(preCon) ? :none : slv.side
    dim = size(opr, 1)

    basVec = similar(inp, dim, 1 + rstItr) # Krylov basis vectors
    #= Every small matrix is held on the CPU in double precision whatever the
    precision of the operator: at fp32 the basis is orthogonal to only ~1e-6, and
    there is no reason to let the small algebra inherit that. =#
    hss = zeros(ComplexF64, 1 + rstItr, rstItr) # Hessenberg matrix
    bMat = zeros(ComplexF64, rcyDim, rstItr) # B = C' * opr * V
    gCof = zeros(ComplexF64, rcyDim) # C' * res, the correction taken in range(U)
    nllSpc = ones(ComplexF64, 1 + rstItr) # Vector in the nullspace of the Hessenberg matrix
    out = isnothing(x0) ? fill!(similar(inp), zero(T)) : copyto!(similar(inp), x0) # Solution vector
    outVec = vec(out)
    buf = similar(outVec) # Work buffer for the restart residual and the cycle update
    # ldiv! is out of place, so what goes into the preconditioner needs its own buffer
    pcnBuf = isnothing(preCon) ? nothing : similar(outVec)
    # U, C, and the buffer a rebuild writes into
    rcyU, rcyC, rcyAlt = ntuple(_ -> similar(inp, dim, rcyDim), 3)
    rcyDot = similar(inp, rcyDim) # C' times a vector, on the device
    nRcy = 0 # Directions of U and C in use
    rcyUv, rcyCv, rcyDotv = view(rcyU, :, 1:nRcy), view(rcyC, :, 1:nRcy), view(rcyDot, 1:nRcy)
    rcyApp = 0 # Applications spent on checking and rebuilding the subspace

    # The preconditioned operator: preCon⁻¹ * opr on the left, opr * preCon⁻¹ on the right
    function oprApp!(dst, src)
        if sid == :left
            mul!(pcnBuf, opr, src)
            ldiv!(dst, preCon, pcnBuf)
        elseif sid == :none
            mul!(dst, opr, src)
        else
            ldiv!(pcnBuf, preCon, src)
            mul!(dst, opr, pcnBuf)
        end
        isnothing(log) || (log.oprApp += 1)
        isnothing(log) || sid == :none || (log.pcnApp += 1)
        return dst
    end
    # Small matrices cross to the device in the precision of the operator
    toDev(sml) = copyto!(similar(inp, T, size(sml)), T.(sml))

    if !isnothing(rcy) && !isnothing(rcy.bas) && rcyDim > 0 &&
        size(rcy.bas, 1) == dim && eltype(rcy.bas) == T
        nRcy = min(size(rcy.bas, 2), rcyDim)
        rcyUv, rcyCv, rcyDotv = view(rcyU, :, 1:nRcy), view(rcyC, :, 1:nRcy), view(rcyDot, 1:nRcy)
        copyto!(rcyUv, view(rcy.bas, :, 1:nRcy))
        copyto!(rcyCv, view(rcy.img, :, 1:nRcy))
        #= A carried C that no longer matches opr * U converges to a confident
        wrong answer, so the pair is probed rather than fingerprinted. An error of
        ε in the invariant caps the attainable residual at ~ε, which is what ties
        the threshold to relTol. The floor is there because a tolerance the
        arithmetic cannot reach would reject every subspace and rebuild forever:
        asking fp32 to verify an invariant to 1e-10 costs rcyDim applications a
        solve and recycles nothing. =#
        prbTol = max(relTol, eps(real(T))^(3//4))
        prb = view(basVec, :, 1) # basVec is idle until the first cycle
        zRnd = toDev(randn(ComplexF64, nRcy))
        mul!(buf, rcyUv, zRnd)
        oprApp!(prb, buf)
        mul!(buf, rcyCv, zRnd)
        prb .-= buf
        rcyApp += 1
        if norm(prb) > prbTol * norm(buf)
            for i in 1:nRcy
                oprApp!(view(rcyC, :, i), view(rcyU, :, i))
            end
            rcyApp += nRcy
            #= CholeskyQR2 on opr * U, one Gram matrix on the device and one
            Cholesky on the host per pass. U / S restores opr * U = C. =#
            sFac = Matrix{ComplexF64}(I, nRcy, nRcy)
            for _ in 1:2
                fac = cholesky!(Hermitian(ComplexF64.(Array(rcyCv' * rcyCv)))).U
                mul!(view(rcyAlt, :, 1:nRcy), rcyCv, toDev(Matrix(inv(fac))))
                copyto!(rcyCv, view(rcyAlt, :, 1:nRcy))
                sFac = fac * sFac
            end
            mul!(view(rcyAlt, :, 1:nRcy), rcyUv, toDev(Matrix(inv(UpperTriangular(sFac)))))
            copyto!(rcyUv, view(rcyAlt, :, 1:nRcy))
        end
    end

    # The first basis vector is the initial residual (preconditioned on the left)
    vk = @view basVec[:, 1]
    if isnothing(x0)
        sid == :left ? ldiv!(vk, preCon, vec(inp)) : copyto!(vk, vec(inp)) # b - opr * x with x = 0
    else
        mul!(buf, opr, outVec)
        isnothing(log) || (log.oprApp += 1)
        if sid == :left
            pcnBuf .= vec(inp) .- buf
            ldiv!(vk, preCon, pcnBuf)
        else
            vk .= vec(inp) .- buf
        end
    end
    isnothing(log) || sid == :left && (log.pcnApp += 1)

    res = norm(vk) # Residual, before the recycled subspace is projected out of it
    #= relTol is measured against the right hand side rather than against the
    initial residual: a good guess shrinks ‖res₀‖, and scaling by it would hold
    a warm start to a tighter absolute threshold than a cold one. The left
    preconditioned residual lives in preCon⁻¹'s image, so the anchor does too. =#
    if isnothing(x0)
        nrmAnc = res
    elseif sid == :left
        ldiv!(pcnBuf, preCon, vec(inp))
        isnothing(log) || (log.pcnApp += 1)
        nrmAnc = norm(pcnBuf)
    else
        nrmAnc = norm(inp)
    end
    absTol = max(absTol, relTol * nrmAnc)
    iszero(nrmAnc) && (nrmAnc = one(nrmAnc))

    # resRec is relative to what the stopping test is anchored to, resTru to ‖inp‖
    nrmRhs = isnothing(log) ? NaN : Float64(nrmRel(inp))
    rstIdx = isnothing(log) ? nothing : Int[] # Iterations a restart fell on
    isnothing(log) || push!(log.resRec, res / nrmAnc)

    β = res
    resAcc = 1.0 # Residual accumulator
    j = 1 # Which step of the cycle we are on
    mCyc = rstItr # Arnoldi steps this cycle has room for
    numItr = 0 # Arnoldi steps taken
    for itr in 1:maxItr
        if res <= absTol || !isfinite(res)
            break
        end
        numItr = itr

        if j == 1
            if nRcy > 0
                #= The optimal correction in range(U) is free, and is folded into
                the update this cycle makes rather than applied on its own. =#
                mul!(rcyDotv, rcyCv', vk)
                view(gCof, 1:nRcy) .= Array(rcyDotv)
                mul!(vk, rcyCv, rcyDotv, -one(T), one(T))
            end
            β = norm(vk)
            iszero(β) || rmul!(vk, inv(β)) # An exact guess leaves no direction to normalize
            resAcc = 1.0
            mCyc = rstItr - nRcy
        end

        vkp1 = @view basVec[:, j + 1]
        oprApp!(vkp1, vk)
        if nRcy > 0
            bCol = view(bMat, 1:nRcy, j)
            fill!(bCol, zero(ComplexF64))
            #= C is rebuilt from the previous C in every cycle, so a loss of
            orthogonality there compounds where V's is discarded at the restart.
            One pass is enough for V and not for C. =#
            for _ in 1:2
                mul!(rcyDotv, rcyCv', vkp1)
                mul!(vkp1, rcyCv, rcyDotv, -one(T), one(T))
                bCol .+= Array(rcyDotv)
            end
        end

        # Orthogonalize vₖ₊₁ against previous basis vectors
        for i in 1:j
            col = @view basVec[:, i]
            hij = dot(col, vkp1)
            hss[i, j] = hij
            axpy!(-hij, col, vkp1) # vkp1 .-= hij .* col
        end

        # Normalize vₖ₊₁ and compute the residual
        hss[j + 1, j] = norm(vkp1)
        #= A vanishing subdiagonal means span(v₁ … vⱼ) is already invariant
        under opr, so the least squares solution over it is exact. vⱼ₊₁ is then
        not a direction at all: it is left as the zero it is rather than
        normalized into a column of NaNs. =#
        if iszero(hss[j + 1, j])
            res = zero(res)
        else
            rmul!(vkp1, inv(T(hss[j + 1, j])))
            nllSpc[j + 1] = -conj(dot(view(nllSpc, 1:j), view(hss, 1:j, j)) / hss[j + 1, j]) # update the nullspace vector
            resAcc += real(abs2(nllSpc[j + 1]))
            res = β / sqrt(resAcc)
        end
        isnothing(log) || push!(log.resRec, res / nrmAnc)

        # Next iteration
        j += 1
        vk = vkp1

        # At the end of cycles (and/or when we have converged/maxed out the iterations), update our solution vector
        don = (res <= absTol || itr == maxItr) # done or not
        if don || j == 1 + mCyc
            stp = j - 1 # Arnoldi steps this cycle took
            wdt = nRcy + stp # Directions the cycle searched over
            #= opr * [U V] = [C V₊] * gBar, taken before lstSqrHss overwrites H.
            Ritz values, harmonic Ritz values and condition estimates all come out
            of it and none of them come back, so it is the matrix that is kept. =#
            gBar = zeros(ComplexF64, wdt + 1, wdt)
            copyto!(view(gBar, 1:nRcy, 1:nRcy), I)
            view(gBar, 1:nRcy, nRcy+1:wdt) .= view(bMat, 1:nRcy, 1:stp)
            view(gBar, nRcy+1:wdt+1, nRcy+1:wdt) .= view(hss, 1:stp+1, 1:stp)
            if !isnothing(log) && (log.hssSmp == :all || isempty(log.hss))
                push!(log.hss, copy(gBar))
            end

            #= Cᴴres is zero after the projection and CᴴV is zero throughout, so
            -B * y annihilates the C component of the residual exactly and the
            problem left to solve is the plain GMRES one, Hy = βe₁. =#
            y = zeros(ComplexF64, stp + 1)
            y[1] = β
            lstSqrHss(view(hss, 1:stp+1, 1:stp), y) # Note: this mutates `hss`, but it's fine because we don't need it anymore
            yDv = toDev(view(y, 1:stp))
            nRcy > 0 && (uDv = toDev(view(gCof, 1:nRcy) .-
                view(bMat, 1:nRcy, 1:stp) * view(y, 1:stp)))

            # Update the solution vector
            if sid == :right
                #= The space on the right is that of opr * precon⁻¹, so what
                the least squares problem returns is precon applied to the
                solution update rather than the update itself. Dividing it back
                once per cycle costs one ldiv! per restart, not one per step. =#
                mul!(buf, view(basVec, :, 1:stp), yDv)
                nRcy > 0 && mul!(buf, rcyUv, uDv, one(T), one(T))
                ldiv!(pcnBuf, preCon, buf)
                isnothing(log) || (log.pcnApp += 1)
                outVec .+= pcnBuf
            else
                mul!(outVec, view(basVec, :, 1:stp), yDv, one(T), one(T))
                nRcy > 0 && mul!(outVec, rcyUv, uDv, one(T), one(T))
            end

            if rcyDim > 0
                # W₊ᴴW, of which only CᴴU and V₊ᴴU are new: CᴴV is zero by the projection
                wpw = zeros(ComplexF64, wdt + 1, wdt)
                copyto!(view(wpw, nRcy+1:wdt, nRcy+1:wdt), I)
                if nRcy > 0
                    view(wpw, 1:nRcy, 1:nRcy) .= Array(rcyCv' * rcyUv)
                    view(wpw, nRcy+1:wdt+1, 1:nRcy) .= Array(view(basVec, :, 1:stp+1)' * rcyUv)
                end
                nLrg = min(round(Int, slv.lrgFrc * rcyDim), wdt)
                nSml = min(rcyDim - nLrg, wdt - nLrg)
                yCof = zeros(ComplexF64, wdt, nLrg + nSml)
                if nSml > 0 # Harmonic Ritz vectors, smallest |θ|
                    egn = eigen(gBar' * gBar, gBar' * wpw)
                    view(yCof, :, 1:nSml) .=
                        view(egn.vectors, :, sortperm(egn.values; by = abs)[1:nSml])
                end
                if nLrg > 0 # Ordinary Ritz vectors, largest |θ|
                    www = Matrix{ComplexF64}(I, wdt, wdt) # WᴴW
                    view(www, nRcy+1:wdt, 1:nRcy) .= view(wpw, nRcy+1:wdt, 1:nRcy)
                    view(www, 1:nRcy, nRcy+1:wdt) .= view(wpw, nRcy+1:wdt, 1:nRcy)'
                    nRcy > 0 && (view(www, 1:nRcy, 1:nRcy) .= Array(rcyUv' * rcyUv))
                    egn = eigen(wpw' * gBar, www)
                    view(yCof, :, nSml+1:nSml+nLrg) .=
                        view(egn.vectors, :, sortperm(egn.values; by = abs, rev = true)[1:nLrg])
                end

                #= The new subspace lives inside the orthonormal [C V₊], so it
                costs two multiplications and no factorization on the device. The
                pivoting is what makes the drop tolerance meaningful, and it drops
                the near duplicates that mixing the two selection rules produces. =#
                fct = qr(gBar * yCof, ColumnNorm())
                nNew = count(i -> abs(fct.R[i, i]) >= sqrt(eps(real(T))) * abs(fct.R[1, 1]),
                    axes(fct.R, 1))
                qCof = toDev(Matrix(fct.Q)[:, 1:nNew])
                zCof = toDev(view(yCof, :, fct.p[1:nNew]) /
                    UpperTriangular(fct.R[1:nNew, 1:nNew]))
                #= U is formed first, so C can be written over the U it replaces.
                U is deliberately left unnormalized: opr * U = C with C
                orthonormal fixes its scaling, and rescaling breaks one or the
                other. The drop above is the lever on conditioning. =#
                mul!(view(rcyAlt, :, 1:nNew), view(basVec, :, 1:stp), view(zCof, nRcy+1:wdt, :))
                nRcy > 0 && mul!(view(rcyAlt, :, 1:nNew), rcyUv, view(zCof, 1:nRcy, :), one(T), one(T))
                mul!(view(rcyU, :, 1:nNew), view(basVec, :, 1:stp+1), view(qCof, nRcy+1:wdt+1, :))
                nRcy > 0 && mul!(view(rcyU, :, 1:nNew), rcyCv, view(qCof, 1:nRcy, :), one(T), one(T))
                rcyU, rcyC, rcyAlt = rcyAlt, rcyU, rcyC
                nRcy = nNew
                rcyUv, rcyCv, rcyDotv = view(rcyU, :, 1:nRcy), view(rcyC, :, 1:nRcy), view(rcyDot, 1:nRcy)
                isnothing(rcy) || (rcy.bas = rcyUv; rcy.img = rcyCv)
                isnothing(log) || push!(log.rcyErr, norm(Array(rcyCv' * rcyCv) - I))
            end

            j = 1

            # If we have not reached max_iter or converged, restart
            if !don
                vk = @view basVec[:, 1]
                mul!(buf, opr, outVec)
                isnothing(log) || (log.oprApp += 1)
                isnothing(log) || push!(rstIdx, itr)
                if sid == :left
                    pcnBuf .= vec(inp) .- buf
                    isnothing(log) || log.truSmp == :never ||
                        push!(log.resTru, norm(pcnBuf) / nrmRhs)
                    ldiv!(vk, preCon, pcnBuf)
                    isnothing(log) || (log.pcnApp += 1)
                else
                    vk .= vec(inp) .- buf
                    isnothing(log) || log.truSmp == :never ||
                        push!(log.resTru, norm(vk) / nrmRhs)
                end
            end
        end
    end
    isnothing(log) || (log.prm = (rstItr = rstItr, maxItr = maxItr, absTol = absTol,
        relTol = relTol, side = slv.side, flexible = false, elmTyp = T,
        arrTyp = Base.typename(typeof(inp)).wrapper, rstIdx = rstIdx,
        rcyDim = rcyDim, lrgFrc = slv.lrgFrc, rcyApp = rcyApp))
    status = !isfinite(res) ? :breakdown : (res <= absTol ? :converged : :maxiter)
    status == :maxiter && @warn "GCRO-DR failed to converge after $maxItr iterations"
    return logEnd!(log, opr, inp, out, status, numItr)
end

"""
    MixPrcRfn{Tlo}

Mixed precision iterative refinement (GMRES-IR): the residual and the solution
update are formed in the precision of the operator, while the correction to the
solution is solved for on a `Tlo` copy of that operator. One high precision
matrix-vector product per outer step buys a high precision backward error at the
matrix-vector cost of `Tlo`.

The operator passed to `solve` must be of higher precision than `Tlo`, and must
match the precision of the right hand side.

Typical usage: `GlaOpr{Float64}(vol, sus; slv=MixPrcRfn(Float32))`

# Fields
- `innSlv::GlaSlv`: The solver used for the low precision corrections
- `cch::Base.RefValue{Any}`: The `(opr, oprLo)` pair of the last solve, rebuilt when the operator changes
- `maxItr::Union{Nothing, Int}`: Maximum number of outer refinement steps (default: 20)
- `absTol::Union{Nothing, Real}`: Absolute tolerance for convergence (default: 0)
- `relTol::Union{Nothing, Real}`: Relative tolerance for convergence (default: √ε)
"""
struct MixPrcRfn{Tlo<:AbstractFloat} <: GlaSlv
    innSlv::GlaSlv # Inner solver for the low precision corrections
    cch::Base.RefValue{Any} # (opr, Tlo copy of opr) of the last solve
    maxItr::Union{Nothing, Int} # Maximum number of outer refinement steps
    absTol::Union{Nothing, Real} # Absolute tolerance
    relTol::Union{Nothing, Real} # Relative tolerance
end

const MixedPrecisionRefinement = MixPrcRfn

"""
    MixPrcRfn(::Type{Tlo}; innSlv::GlaSlv=GMRESSolver(), maxItr=nothing, absTol=nothing, relTol=nothing)

Create a `MixPrcRfn` refining in the precision `Tlo`. Unset tolerances are
resolved against the right hand side at solve time.

# Arguments
- `Tlo::Type{<:AbstractFloat}`: The precision of the correction solves
- `innSlv::GlaSlv=GMRESSolver()`: The solver for the correction solves
- `maxItr`, `absTol`, `relTol`: Outer loop settings, resolved at solve time when left unset

# Returns
- `MixPrcRfn{Tlo}`: A new solver instance holding no operator copy yet
"""
MixPrcRfn(::Type{Tlo}; innSlv::GlaSlv=GMRESSolver(), maxItr=nothing, absTol=nothing,
    relTol=nothing) where Tlo<:AbstractFloat =
    MixPrcRfn{Tlo}(innSlv, Ref{Any}((nothing, nothing)), maxItr, absTol, relTol)

# Solver settings with the unset ones filled in from the right hand side
slvPrm(slv::MixPrcRfn, vec::AbstractVector) = (
    maxItr = @something(slv.maxItr, 20),
    absTol = @something(slv.absTol, zero(real(eltype(vec)))),
    relTol = @something(slv.relTol, sqrt(eps(real(eltype(vec))))))

"""
    solve(opr::AbstractGlaOpr{Thi}, inp::AbstractVector{Complex{Thi}}, slv::MixPrcRfn{Tlo}; log = nothing) where {Thi, Tlo}

Solve the linear system `opr * out = inp` by iterative refinement, with
corrections computed in the precision `Tlo`.

# Arguments
- `opr::AbstractGlaOpr{Thi}`: The operator in the linear system, of precision `Thi`
- `inp::AbstractVector{Complex{Thi}}`: The right hand side vector
- `slv::MixPrcRfn{Tlo}`: The solver parameters
- `log`: A `SlvLog` to fill in place, or `nothing` (default) for no record. One
  inner record per refinement step lands in `log.inner`

# Returns
- `out::AbstractVector{Complex{Thi}}`: The solution vector

# Notes
- The `Tlo` copy of the operator is cached and reused across solves with the
  same operator, and rebuilt as soon as a different operator is passed
- Refinement in the precision of the operator is pointless, and throws
"""
function solve(opr::AbstractGlaOpr{Thi}, inp::AbstractVector{Complex{Thi}},
    slv::MixPrcRfn{Tlo}; log = nothing) where {Thi<:AbstractFloat, Tlo<:AbstractFloat}
    if Tlo == Thi
        throw(ArgumentError("Iterative refinement in the precision it refines from does nothing: both the correction solves and the right hand side are $Thi. Either drop MixPrcRfn or raise the precision of the operator and the right hand side."))
    end
    (; maxItr, absTol, relTol) = slvPrm(slv, inp)
    if first(slv.cch[]) !== opr
        slv.cch[] = (opr, Base.typename(typeof(opr)).wrapper{Tlo}(opr))
    end
    oprLo = last(slv.cch[])
    #= The cached copy has to track every in place change to the operator it was
    built from: adjoint! flips the adjoint mode, setSus! replaces the
    susceptibility, and both leave the copy behind. =#
    isadjoint(oprLo) != isadjoint(opr) && (oprLo = adjoint!(oprLo))
    hasproperty(opr, :sus) && (oprLo.sus = Base.typename(typeof(opr.sus)).wrapper{Tlo}(opr.sus))
    slv.cch[] = (opr, oprLo)

    out = zero(inp) # Solution vector
    res = copy(inp) # Residual
    buf = similar(inp) # Work buffer for the high precision matrix-vector product
    #= The inner operator is the same in every refinement step and only the right
    hand side moves, so a recycled subspace carries across the outer loop. =#
    rcy = slv.innSlv isa GCRODRSolver ? RcySpc() : nothing
    absTol = max(absTol, relTol * norm(inp))
    nrmRhs = nrmRel(inp)
    isnothing(log) || (log.prm = (maxItr = maxItr, absTol = absTol, relTol = relTol,
        elmTyp = Complex{Thi}, prcRfn = Tlo,
        arrTyp = Base.typename(typeof(inp)).wrapper))

    for stp in 1:maxItr
        nrmRes = norm(res)
        isnothing(log) || push!(log.resRec, nrmRes / nrmRhs)
        if nrmRes <= absTol
            return logEnd!(log, opr, inp, out, :converged, stp - 1)
        end
        innLog = isnothing(log) ? nothing : SlvLog(; truSmp = log.truSmp, hssSmp = log.hssSmp)
        # Normalized before narrowing, so the inner right hand side is O(1)
        rhsLo = Complex{Tlo}.(res ./ nrmRes)
        dirLo = isnothing(rcy) ? solve(oprLo, rhsLo, slv.innSlv; log = innLog) :
            solve(oprLo, rhsLo, slv.innSlv; log = innLog, rcy)
        isnothing(log) || push!(log.inner, innLog)
        out .+= nrmRes .* dirLo
        mul!(buf, opr, out)
        isnothing(log) || (log.oprApp += 1)
        res .= inp .- buf
    end
    if norm(res) > absTol
        @warn "Iterative refinement did not converge after $maxItr iterations."
        return logEnd!(log, opr, inp, out, :maxiter, maxItr)
    end
    return logEnd!(log, opr, inp, out, :converged, maxItr)
end

"""
    lstSqrHss(hss::AbstractMatrix{T}, inp::AbstractVector{T}) where T

Solve the least squares problem for the Hessenberg matrix in GMRES using
Givens rotations.

# Arguments
- `hss::AbstractMatrix{T}`: The Hessenberg matrix
- `inp::AbstractVector{T}`: The right-hand side vector

# Returns
- `inp::AbstractVector{T}`: The solution vector (mutated in-place)

# Notes
- This function uses Givens rotations to transform the Hessenberg matrix into
  upper triangular form
- The solution is computed by back substitution
"""
function lstSqrHss(hss::AbstractMatrix{T}, inp::AbstractVector{T}) where T
    # Yoinked from https://github.com/JuliaLinearAlgebra/IterativeSolvers.jl/blob/0b2f1c5d352069df1bc891750087deda2d14cc9d/src/hessenberg.jl

    wdt = size(hss, 2) # width

    for i in 1:wdt
        c, s, _ = LinearAlgebra.givensAlgorithm(hss[i, i], hss[i + 1, i])
        hss[i, i] = c * hss[i, i] + s * hss[i + 1, i]
        for j in i+1:wdt
            temp = -conj(s) * hss[i, j] + c * hss[i + 1, j]
            hss[i, j] = c * hss[i, j] + s * hss[i + 1, j]
            hss[i + 1, j] = temp
        end
        tmp = -conj(s) * inp[i] + c * inp[i + 1]
        inp[i] = c * inp[i] + s * inp[i + 1]
        inp[i + 1] = tmp
    end

    U = UpperTriangular(view(hss, 1:wdt, 1:wdt))
    ldiv!(U, view(inp, 1:wdt))

    return inp
end
