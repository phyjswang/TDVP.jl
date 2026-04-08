using TensorOperations
using LinearAlgebra
using MatrixAlgebraKit
using KrylovKit


function normalizeA(A::Array{T,3}) where T<:Union{ComplexF64,Float64}
    D = size(A,1)
    vals, _, _ = eigsolve(
        x -> (@tensor xnew[:] := A[-1,1,2] * x[2,3] * conj(A)[-2,1,3]),
        rand(ComplexF64,D,D),
        1,
        :LM,
        Arnoldi()
    )
    abs(imag(vals[1])) > 1e-8 && error("val must be real!")
    return A/√vals[1], vals[1]
end

# AL = L A L^-1
function leftOrthonormalize(A::Array{ComplexF64,3}; η::Float64 = 1e-8, imax::Int64 = 1000, verbose::Int64 = 0)
    d = size(A,2)
    D = size(A,1)
    D ≠ size(A,3) && error("left and right D must be equal")

    L = normalize(rand(ComplexF64,D,D))
    Lold = L
    @tensor LA[:] := L[-1,1] * A[1,-2,-3]
    AL, L = qr_compact(reshape(LA,D*d,D); positive = true)
    AL = reshape(AL,D,d,D)
    λ = norm(L)
    normalize!(L)
    δ = norm(L - Lold)

    i = 0
    while δ > η
        i += 1
        Lold = L
        @tensor LA[:] := L[-1,1] * A[1,-2,-3]
        AL, L = qr_compact(reshape(LA,D*d,D); positive = true)
        AL = reshape(AL,D,d,D)
        λ = norm(L)
        normalize!(L)
        δ = norm(L - Lold)

        verbose > 0 && println("i = $i, δ = $δ")
        i > imax && break
    end

    i > imax && println("reach maximal iter!")

    return AL, L, λ
end

# AR = R^-1 A R
function rightOrthonormalize(A::Array{ComplexF64,3}; η::Float64 = 1e-8, imax::Int64 = 1000, verbose::Int64 = 0)
    d = size(A,2)
    D = size(A,1)
    D ≠ size(A,3) && error("left and right D must be equal")

    R = normalize(rand(ComplexF64,D,D))
    Rold = R
    @tensor AR[:] := A[-1,-2,1] * R[1,-3]
    R, A_R = lq_compact(reshape(AR,D,d*D); positive = true)
    A_R = reshape(A_R,D,d,D)
    λ = norm(R)
    normalize!(R)
    δ = norm(R - Rold)

    i = 0
    while δ > η
        i += 1
        Rold = R
        @tensor AR[:] := A[-1,-2,1] * R[1,-3]
        R, A_R = lq_compact(reshape(AR,D,d*D); positive = true)
        A_R = reshape(A_R,D,d,D)
        λ = norm(R)
        normalize!(R)
        δ = norm(R - Rold)

        verbose > 0 && println("i = $i, δ = $δ")
        i > imax && break
    end

    i > imax && println("reach maximal iter!")

    return A_R, R, λ
end

function get_MixedGauge(A::Array{T,3}; η::Float64 = 1e-8, verbose::Int64 = 0) where T<:Union{ComplexF64,Float64}
    AL, _, λ = leftOrthonormalize(A;η=η)
    AR, C, λ1 = rightOrthonormalize(AL;η=η)
    verbose > 0 && println("rel. diff. betw. 2 λs = ",abs((λ - λ1)/λ))
    normalize!(C)
    # U, C, Vdag = svd_trunc(C; trunc = (atol = eps(), rtol = eps(), maxerror = eps()))
    # @tensor AL[:] := U'[-1,1] * AL[1,-2,2] * U[2,-3]
    # @tensor AR[:] := Vdag[-1,1] * AR[1,-2,2] * Vdag'[2,-3]
    return AL, AR, C, λ
end
