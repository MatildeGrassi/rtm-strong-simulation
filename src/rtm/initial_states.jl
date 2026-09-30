# initial_states.jl — boundary temporal MPS at the spatial edges (RTM).

module InitialStates

using ITensors, ITensorMPS

export product_of_bellpairs_with_zeros_ends_MPS

# Unnormalised Bell creator: |00⟩ → |00⟩ + |11⟩ (basis |00⟩, |01⟩, |10⟩, |11⟩).
function bell_phi_plus_creator_4x4()::Matrix{ComplexF64}
    s::Int = 1
    return ComplexF64[
        s  0  0  s;
        0  1  0  0;
        0  0  1  0;
        s  0  0 -s
    ]
end

# 4×4 matrix → ITensor on (i, j): rows = output (primed), columns = input.
function two_site_gate_from_mat(U4::Matrix{ComplexF64}, i::Index, j::Index)::ITensor
    ip::Index = prime(i)
    jp::Index = prime(j)
    G::ITensor = ITensor(i, j, ip, jp)
    for a in 1:2, b in 1:2, ap in 1:2, bp in 1:2
        out::Int = (ap-1)*2 + bp
        inn::Int = (a -1)*2 + b
        G[i=>a, j=>b, ip=>ap, jp=>bp] = U4[out, inn]
    end
    return G
end

# Edge tMPS: site 1 = x bit, site N = y bit, Bell pairs on (2,3), (4,5), …
# (they join the two legs of each time slice at the open spatial edge).
function product_of_bellpairs_with_zeros_ends_MPS(
        sites::Vector{<:Index}, first_bitstring::Int, first_bitstring_y::Int)::MPS
    N::Int = length(sites)
    @assert iseven(N) "Need even N"
    @assert N ≥ 4    "Need N ≥ 4 to have at least one Bell pair plus fixed ends"

    psi::MPS = productMPS(sites, "0")

    if first_bitstring == 1
        psi[1] = psi[1] * op("X", sites[1])
        psi[1] = noprime(psi[1])
    end
    if first_bitstring_y == 1
        psi[N] = psi[N] * op("X", sites[N])
        psi[N] = noprime(psi[N])
    end

    U4::Matrix{ComplexF64} = bell_phi_plus_creator_4x4()
    gates::Vector{ITensor} = ITensor[]
    for n in 2:2:N-2
        push!(gates, two_site_gate_from_mat(U4, sites[n], sites[n+1]))
    end
    psi = apply(gates, psi; cutoff=0.0, maxdim=4)
    return psi
end

end
