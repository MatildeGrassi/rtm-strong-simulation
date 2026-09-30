# random_gates.jl — random two-qubit gates for the RTM code. tebd_fw.jl and tebd_tr.jl hold
# copies with identical arithmetic and draw order, so all three build the same circuit.

module RandomGates

using LinearAlgebra, Random

export random_unitary_4x4, random_circuit_gates

const SWAP4 = ComplexF64[1 0 0 0;
                         0 0 1 0;
                         0 1 0 0;
                         0 0 0 1]

# Haar U0 (QR with phase fix); symmetric = true → polar part of (U0 + SWAP U0 SWAP)/2.
# Both ensembles use the same random numbers.
function random_unitary_4x4(rng::AbstractRNG; symmetric::Bool=true)::Matrix{ComplexF64}
    Q, R = qr(randn(rng, ComplexF64, 4, 4))
    U0::Matrix{ComplexF64} = Matrix(Q) * Diagonal(sign.(diag(R)))
    symmetric || return U0
    F = svd(0.5 * (U0 + SWAP4 * U0 * SWAP4))
    return F.U * F.Vt
end

# One shared gate, or a Dict (t, layer, bond) drawn per period: odd bonds, then even bonds.
function random_circuit_gates(rng::AbstractRNG, n_sites::Int, steps::Int,
                              translational_invariance::Bool; symmetric::Bool=true)
    translational_invariance && return random_unitary_4x4(rng; symmetric=symmetric)
    gates = Dict{Tuple{Int,Int,Int}, Matrix{ComplexF64}}()
    for t in 1:steps, layer in 1:2, i in layer:2:n_sites-1
        gates[(t, layer, i)] = random_unitary_4x4(rng; symmetric=symmetric)
    end
    return gates
end

end
