# entanglement.jl — quantities of an RTM singular-value spectrum. The weights are linear,
# p_i = σ_i / Σσ (the RTM is not a density matrix), so these are not von Neumann entropies.

module Entanglement

using ITensors

export entropy_from_singular_values, sumsq_singular_values, renyi2_from_singular_values

# Shannon entropy -Σ p log p of p_i = σ_i / Σσ, from the diagonal ITensor S of an SVD.
function entropy_from_singular_values(S::ITensor)::Float64
    eps::Float64   = 1.0e-14
    probs::Vector{Float64} = Float64[]
    total::Float64 = 0.0
    for n in 1:dim(S, 1)
        s::Float64 = abs(S[n, n])
        push!(probs, s)
        total += s
    end
    total < eps && return 0.0
    SvN::Float64 = 0.0
    for p in probs
        pn::Float64 = p / total
        pn > eps && (SvN -= pn * log(pn))
    end
    return SvN
end

# Σ σ_i² (a norm, not an entropy; depends on the boundary normalisation).
sumsq_singular_values(sv::AbstractVector{<:Real})::Float64 = sum(abs2, sv)

# Rényi-2 entropy -log Σ p_i² of p_i = σ_i / Σσ (scale invariant; 0 for an empty spectrum).
function renyi2_from_singular_values(sv::AbstractVector{<:Real})::Float64
    s::Vector{Float64}  = abs.(sv)
    total::Float64      = sum(s)
    total <= 0.0 && return 0.0
    p::Vector{Float64}  = s ./ total
    return -log(sum(abs2, p))
end

end
