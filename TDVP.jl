using TensorOperations
using LinearAlgebra
using MatrixAlgebraKit
using KrylovKit

include("mixedGauge.jl")

# basic tensors
# -------------
const sx = [0. 1.; 1. 0.] ./ 2
const sz = [1. 0.; 0. -1.] ./ 2
const id = [1. 0.; 0. 1.]

function getHLHR(AL::Array{ComplexF64,3}, AR::Array{ComplexF64,3}, HL::Matrix{ComplexF64}, HR::Matrix{ComplexF64}, h2::Array{ComplexF64, 4}; εS::Float64 = 1e-12)
    @tensor hL[:] := AL[1,2,3] * AL[3,4,-1] * h2[5,6,2,4] * conj(AL)[1,5,7] * conj(AL)[7,6,-2]
    @tensor hR[:] := AR[-1,2,1] * AR[1,3,4] * h2[6,5,2,3] * conj(AR)[7,5,4] * conj(AR)[-2,6,7]

    vals, vecs, _ = eigsolve(
        x -> (@tensor xnew[:] := AL[-1,2,1] * x[1,3] * conj(AL)[-2,2,3]),
        rand(ComplexF64,size(AL,1),size(AL,1)),
        1,
        :LM,
        Arnoldi()
    )

    R = transpose(vecs[1])
    R /= tr(R)

    vals, vecs, _ = eigsolve(
        x -> (@tensor xnew[:] := AR[1,3,-1] * x[1,2] * conj(AR)[2,3,-2]),
        rand(ComplexF64,size(AR,3),size(AR,3)),
        1,
        :LM,
        Arnoldi()
    )

    L = transpose(vecs[1])
    L /= tr(L)

    t1 = Matrix{ComplexF64}(I, size(AL,1), size(AL,1))

    e1 = tr(hL * R)
    e2 = tr(L * hR)

    HL, _ = linsolve(
        HL -> HL - (@tensor HLTL[:] := AL[1,3,-1] * HL[1,2] * conj(AL)[2,3,-2]) + tr(HL * R) * t1,
        hL - e1 * t1,
        HL,
        GMRES(tol = εS)
    )

    HR, _ = linsolve(
        HR -> HR - (@tensor TRHR[:] := AR[-1,2,1] * HR[1,3] * conj(AR)[-2,2,3]) + tr(L * HR) * t1,
        hR - e2 * t1,
        HR,
        GMRES(tol = εS)
    )

    return HL, HR
end

function getAC(h2::Array{ComplexF64, 4}, HL::Matrix{ComplexF64}, HR::Matrix{ComplexF64}, AL::Array{ComplexF64,3}, AR::Array{ComplexF64,3}, AC::Array{ComplexF64,3})
    @tensor AC′[:] := AL[4,2,1] * AC[1,3,-3] * h2[5,-2,2,3] * conj(AL)[4,5,-1]
    AC′ += (@tensor rslt[:] := AC[-1,2,1] * AR[1,3,4] * h2[-2,5,2,3] * conj(AR)[-3,5,4])
    AC′ += tensorcontract(HL, (1,-1), AC, (1,-2,-3))
    AC′ += tensorcontract(AC, (-1,-2,1), HR, (1,-3))
    return AC′
end

function getC(h2::Array{ComplexF64, 4}, HL::Matrix{ComplexF64}, HR::Matrix{ComplexF64}, AL::Array{ComplexF64,3}, AR::Array{ComplexF64,3}, C::Matrix{ComplexF64})
    @tensor C′[:] := AL[5,3,1] * C[1,2] * AR[2,4,7] * h2[6,8,3,4] * conj(AL)[5,6,-1] * conj(AR)[-2,8,7]
    C′ += tensorcontract(HL, (1,-1), C, (1,-2))
    C′ += tensorcontract(C, (-1,1), HR, (1,-2))
    return C′
end

function getALAR(AC::Array{ComplexF64,3}, C::Matrix{ComplexF64})
    WAC, _ = left_polar(reshape(AC,:,size(AC,3)))
    WC,  _  = left_polar(C)
    AL = reshape(WAC * WC', size(AC,1), size(AC,2), size(AC,3))
    _, WAC = right_polar(reshape(AC,size(AC,1),:))
    _, WC  = right_polar(C)
    AR = reshape(WC' * WAC, size(AC,1), size(AC,2), size(AC,3))
    return AL, AR
end

function VUMPS(g::Float64; D::Int64 = 16, ε::Float64 = 1e-12, imax::Int64 = 1000, εratio = 1e-1)
    @tensor h2[:] := -2 * sz[-1,-3] * sz[-2,-4] + g/2 * sx[-1,-3] * id[-2,-4] + g/2 * id[-1,-3] * sx[-2,-4]

    h2 = ComplexF64.(h2)

    C = ComplexF64.(Matrix(I, D, D))
    AC = randn(ComplexF64, D, 2, D)
    normalize!(C)
    AL, AR = getALAR(AC, C)
    AC = tensorcontract(AL, (-1,-2,1), C, (1,-3))
    HL = rand(ComplexF64, size(AL,1),size(AL,1))
    HR = rand(ComplexF64, size(AR,3),size(AR,3))

    εprec = .42

    i = 0
    while εprec > ε
        i += 1
        HL, HR = getHLHR(AL, AR, HL, HR, h2;  εS = εprec * εratio)
        _, vecsAC, _ = eigsolve(
            x -> getAC(h2, HL, HR, AL, AR, x),
            AC,
            1,
            :SR,
            Arnoldi(tol = εprec * εratio)

        )
        _, vecsC, _ = eigsolve(
            x -> getC(h2, HL, HR, AL, AR, x),
            C,
            1,
            :SR,
            Arnoldi(tol = εprec * εratio)
        )
        AC = vecsAC[1]
        C = vecsC[1]
        AL, AR = getALAR(AC, C)

        εL = norm(tensorcontract(AL, (-1,-2,1), C, (1,-3)) - AC)
        εR = norm(tensorcontract(C, (-1,1), AR, (1,-2,-3)) - AC)

        εprec = max(εL,εR)

        i > imax && break
    end

    if i > imax
        println("Not converged at i = ",i)
    else
        println("Converged at i = ",i)
    end

    return AC, C, AL, AR
end

function tdvp(h2::Array{ComplexF64, 4}, AC::Array{ComplexF64,3}, C::Matrix{ComplexF64}, AL::Array{ComplexF64,3}, AR::Array{ComplexF64,3}, HL::Matrix{ComplexF64}, HR::Matrix{ComplexF64}, z::Number; εG::Float64 = 1e-8, isreorno::Bool = true)
    HL, HR = getHLHR(AL, AR, HL, HR, h2)
    AC, _ = exponentiate(
        x -> getAC(h2, HL, HR, AL, AR, x),
        -z,
        AC,
        Arnoldi()
    )
    C, _ = exponentiate(
        x -> getC(h2, HL, HR, AL, AR, x),
        -z,
        C,
        Arnoldi()
    )

    AL, AR = getALAR(AC, C)

    # reorthonormalize
    if isreorno
        εL = norm(tensorcontract(AL, (-1,-2,1), C, (1,-3)) - AC)
        εR = norm(tensorcontract(C, (-1,1), AR, (1,-2,-3)) - AC)
        if max(εL,εR) > εG
            println("reorthonormalize")
            AL, AR, C, _ = get_MixedGauge(AL; η = 1e-12)
            @tensor AC[:] := AL[-1,-2,1] * C[1,-3]
            HL, HR = getHLHR(AL, AR, HL, HR, h2)
        end
    end

    return AC, C, AL, AR, HL, HR
end

function getλs(AL::Array{ComplexF64,3})
    vals, _, _ = eigsolve(
        x -> (@tensor rslt[:] := AL[-1,2,1] * x[1,3] * AL[-2,2,3]),
        rand(ComplexF64, size(AL,3), size(AL,1)),
        1,
        :LM,
        Arnoldi()
    )
    return abs(vals[1])
end

function main(; tmax::Float64, g1::Float64 = 0.5, D::Int64 = 64, g0::Float64 = 0.2, dz::Float64)
    @tensor h2[:] := -2 * sz[-1,-3] * sz[-2,-4] + g1/2 * sx[-1,-3] * id[-2,-4] + g1/2 * id[-1,-3] * sx[-2,-4]
    h2 = ComplexF64.(h2)

    AC, C, AL, AR = VUMPS(g0; D)
    @show size(C,1)

    HL = rand(ComplexF64, size(AL,1),size(AL,1))
    HR = rand(ComplexF64, size(AR,3),size(AR,3))
    HL, HR = getHLHR(AL, AR, HL, HR, h2)

    nz = Int64(tmax / dz / 2)
    iz = 0

    lsλ1 = []
    lsz = []
    z = 0

    while iz < nz
        iz += 1
        z += dz * 2
        iz % 10 == 1 && println("iz / nz = ", iz, " / ", nz, ", D = ", size(C,1))

        AC, C, AL, AR, HL, HR = tdvp(h2, AC, C, AL, AR, HL, HR, dz * 1.0im)

        λ1 = getλs(AL)
        push!(lsz, z)
        push!(lsλ1, λ1)
    end

    return lsλ1, lsz
end

g0 = 1.5
g1 = 0.2
tmax = 2.0
D = 16
dz = 0.002

lsλ1, lsz = main(; tmax, g1, D, g0, dz)

# plot
# ----
using CairoMakie
f=Figure(
    size = (1,1) .* 300
)
ax=Axis(
    f[1,1],
    xlabel = L"t",
    ylabel = "Loschmidt rate function"
)

lines!(
    ax,
    abs.(lsz),
    -2 * log.(lsλ1),
    label = "TDVP"
)

include("exact.jl")

lst = 0:0.005:(tmax)
lines!(
    ax,
    lst,
    [real(rf(1.0im * t, 1/2, g0, g1)) * 2 for t in lst],
    linestyle = :dash,
    color = :red,
    label = "Exact"
)

axislegend(ax,position = :lt)

f

save("result.pdf", f)
