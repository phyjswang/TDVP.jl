using QuadGK

ε(J::Float64, g::Float64,k::Float64) = 2*J * √((g-cos(k))^2 + sin(k)^2)

function θ(g::Float64,k::Float64)
    θ1 = atan(sin(k) / (g-cos(k))) / 2
    if θ1 < 0
        return θ1 + π/2
    else
        return θ1
    end
end

φ(g0::Float64,g1::Float64,k::Float64) = θ(g0,k) - θ(g1,k)

rf(z::ComplexF64,J::Float64, g0::Float64,g1::Float64) = quadgk(k -> - 1 / 2π * log(cos(φ(g0,g1,k))^2 + sin(φ(g0,g1,k))^2 * exp(-2*z*ε(J,g1,k))), 0, π)[1]
