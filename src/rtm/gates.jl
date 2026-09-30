# gates.jl — two-qubit gates rotated for the transverse contraction (RTM).
# Builds one rotated layer per spatial bond on the temporal chain, and applies layers with
# the boundary operations on the x / y bits (sites 1 and N).

module Gates

using ITensors, ITensorMPS, LinearAlgebra
include("tensor_utils.jl")
using .TensorUtils: PAULI_MATRICES

export make_bond_layers, apply_odd_layer!, apply_even_layer!

# Trotter gate exp(-i dt H_bond) of the Ising chain, H_bond = J XX + aL H1⊗I + aR I⊗H1,
# H1 = hx X + hz Z. The field is split in half between bonds; an edge site takes it whole.
function twoqubit_floquet_gate_layer(dt::Real; J::Real, hx::Real, hz::Real, layer::Int, i::Int, N::Int, boundary_L::Bool,boundary_R::Bool)
    @assert layer == 1 || layer == 2
    @assert 1 ≤ i ≤ N-1
    (; Id2, σx, σz) = PAULI_MATRICES

    H1 = hx*σx + hz*σz
    XX = kron(σx, σx)

    aL::Float64 = boundary_L ? 1.0 : 0.5
    aR::Float64 = boundary_R ? 1.0 : 0.5

    Hbond = J*XX + aL*kron(H1, Id2) + aR*kron(Id2, H1)
    return exp(-1im * dt * Hbond)
end

# Kicked-Ising gate: exp(-i[XX + g(XI+IX)/2]) exp(-i h(ZI+IZ)/2), g = 0.81, h = 0.904508.
# dt, J, layer, i, N are not used by the gate itself.
function transverse_field_floquet_ising_gate(dt::Float64; J::Float64, layer::Int,
                                              i::Int, N::Int)::Matrix{ComplexF64}
    @assert layer == 1 || layer == 2
    @assert 1 ≤ i ≤ N-1
    (; Id2, σx, σz) = PAULI_MATRICES
    XX::Matrix{ComplexF64}    = kron(σx, σx)
    ZI::Matrix{ComplexF64}    = kron(σz, Id2)
    IZ::Matrix{ComplexF64}    = kron(Id2, σz)
    XI::Matrix{ComplexF64}    = kron(σx, Id2)
    IX::Matrix{ComplexF64}    = kron(Id2, σx)
    Hbond::Matrix{ComplexF64} = XX + 0.81 * (XI + IX)/2
    Hbond2::Matrix{ComplexF64}= 0.904508 * (ZI + IZ)/2
    return exp(-1im * Hbond) * exp(-1im * Hbond2)
end

# 4×4 matrix → rotated gate on temporal sites (i1, i2): columns ↔ (i1, i1'), rows ↔ (i2, i2').
# Unprimed legs = left spatial site, primed legs = right spatial site.
function makeNonUnitaryGate(RandUnitary::Matrix{ComplexF64},
                             i1::Index, i2::Index)::ITensor
    i1p::Index = prime(i1)
    i2p::Index = prime(i2)
    G::ITensor = ITensor(i1, i1p, i2, i2p)
    for row in 1:4
        i2val::Int  = (row-1) ÷ 2 + 1
        i2pval::Int = (row-1) % 2 + 1
        for col in 1:4
            i1val::Int  = (col-1) ÷ 2 + 1
            i1pval::Int = (col-1) % 2 + 1
            G[i1 => i1val, i1p => i1pval,
              i2 => i2val, i2p => i2pval] = RandUnitary[row, col]
        end
    end
    return G
end

# Rotated gates of one spatial bond on the temporal chain (bulk, left-edge, right-edge
# versions; they differ only for trotter). Layer 1 → temporal bonds 1, 3, …, 2T+1 (bond 1 is
# the identity), layer 2 → 2, 4, …, 2T. Temporal bonds run from period T (top) down to 1.
function MakeGates(sites, n_sites, dt; J, hx, hz, layer::Int, gate_random::Bool=false,
                    U4_shared_random=nothing, random_gates=nothing,
                    translational_invariance::Bool=true, spatial_bond::Int=1, circuit_steps::Int=1,
                    trotter::Bool=false)
    (; Id4) = PAULI_MATRICES
    Gates = ITensor[]
    Gates_boundary_L = ITensor[]
    Gates_boundary_R = ITensor[]

    use_circuit::Bool = !trotter

    for i in 2-(layer % 2):2:n_sites-1
        if layer == 1 && i == 1
            U4 = Id4
            U4_boundary_L = U4
            U4_boundary_R = U4
        else
            if use_circuit
                if gate_random
                    if translational_invariance
                        U4=U4_shared_random
                        U4_boundary_L = U4
                        U4_boundary_R = U4
                    else
                        # temporal bond i → period t of the forward circuit
                        time_step_from_top = layer == 1 ? (i - 1) ÷ 2 : i ÷ 2
                        time_step = circuit_steps - time_step_from_top + 1
                        U4=random_gates[(time_step, layer, spatial_bond)]
                        U4_boundary_L = U4
                        U4_boundary_R = U4
                    end
                else
                    U4=transverse_field_floquet_ising_gate(dt; J=J, layer=layer, i=i, N=n_sites)
                    U4_boundary_L = U4
                    U4_boundary_R = U4
                end
            else
                U4 = twoqubit_floquet_gate_layer(dt; J=J, hx=hx, hz=hz, layer=layer, i=i, N=n_sites, boundary_L=false, boundary_R=false)
                U4_boundary_L = twoqubit_floquet_gate_layer(dt; J=J, hx=hx, hz=hz, layer=layer, i=i, N=n_sites, boundary_L=true, boundary_R=false)
                U4_boundary_R = twoqubit_floquet_gate_layer(dt; J=J, hx=hx, hz=hz, layer=layer, i=i, N=n_sites, boundary_L=false, boundary_R=true)
            end
        end
        # transpose to match the index convention of the forward gate (checked against FW)
        push!(Gates, makeNonUnitaryGate(permutedims(U4), sites[i], sites[i+1]))
        push!(Gates_boundary_L, makeNonUnitaryGate(permutedims(U4_boundary_L), sites[i], sites[i+1]))
        push!(Gates_boundary_R, makeNonUnitaryGate(permutedims(U4_boundary_R), sites[i], sites[i+1]))
    end
    return Gates, Gates_boundary_L, Gates_boundary_R
end

# One rotated layer per spatial bond b = 1 … L-1 (odd b → layer 1, even b → layer 2).
# GatesL[b] evolves psiL (left → right); bonds 1 and L-1 get the edge gates.
# GatesR[b] evolves psiR (right → left): prime levels swapped = rotated SWAP·U·SWAP,
# needed for gates that do not commute with SWAP (Haar, trotter edges).
function make_bond_layers(sites, n_sites::Int, n_sites_fw::Int, steps_fw::Int, dt::Float64;
                          J, hx, hz, gate_random::Bool, random_gates,
                          translational_invariance::Bool, trotter::Bool=false)
    n_bonds::Int   = n_sites_fw - 1
    per_bond::Bool = gate_random && !translational_invariance
    GatesL = Vector{Vector{ITensor}}(undef, n_bonds)
    GatesR = Vector{Vector{ITensor}}(undef, n_bonds)
    for b in 1:n_bonds
        G, G_bL, G_bR = MakeGates(sites, n_sites, dt; J=J, hx=hx, hz=hz,
                                  layer = isodd(b) ? 1 : 2, gate_random=gate_random,
                                  U4_shared_random = per_bond ? nothing : random_gates,
                                  random_gates     = per_bond ? random_gates : nothing,
                                  translational_invariance=translational_invariance,
                                  spatial_bond=b, circuit_steps=steps_fw,
                                  trotter=trotter)
        GatesL[b] = b == 1 ? G_bL : (b == n_bonds ? G_bR : G)
        GatesR[b] = [swapprime(g, 0 => 1) for g in GatesL[b]]
    end
    return GatesL, GatesR
end

# Odd layer, then project site 1 on the x bit and site N on the y bit ((I ± Z)/2).
# Returns the normalised MPS and the log of the norm removed.
function apply_odd_layer!(psi_tmp::MPS,
                          GatesOdd::Vector{ITensor},
                          maxbond::Int,
                          cutoff::Union{Float64, Nothing},
                          bit_val_out::Int,
                          bit_val_out_y::Int,
                          Id1::ITensor, Sz1::ITensor,
                          IdN::ITensor, SzN::ITensor)::Tuple{MPS, Float64}
    psi_tmp = apply(GatesOdd, psi_tmp; maxdim=maxbond, cutoff=cutoff)

    if bit_val_out == 1
        psi_tmp[1] = psi_tmp[1] * (Id1 - Sz1) / 2
    else
        psi_tmp[1] = psi_tmp[1] * (Id1 + Sz1) / 2
    end
    if bit_val_out_y == 1
        psi_tmp[length(psi_tmp)] = psi_tmp[length(psi_tmp)] * (IdN - SzN) / 2
    else
        psi_tmp[length(psi_tmp)] = psi_tmp[length(psi_tmp)] * (IdN + SzN) / 2
    end
    psi_tmp[1]               = noprime(psi_tmp[1])
    psi_tmp[length(psi_tmp)] = noprime(psi_tmp[length(psi_tmp)])

    curr_norm::Float64 = norm(psi_tmp)
    log_gain::Float64  = log(curr_norm)
    normalize!(psi_tmp)
    return psi_tmp, log_gain
end

# Even layer, then flip site 1 / N (X) when the next x / y bit differs from the current one.
# Returns the normalised MPS and the log of the norm removed.
function apply_even_layer!(psi_tmp::MPS,
                           GatesEven::Vector{ITensor},
                           maxbond::Int,
                           cutoff::Union{Float64, Nothing},
                           bit_val_in::Int,
                           bit_val_out::Int,
                           bit_val_in_y::Int,
                           bit_val_out_y::Int,
                           SitesInit::Vector{<:Index})::Tuple{MPS, Float64}
    psi_tmp = apply(GatesEven, psi_tmp; maxdim=maxbond, cutoff=cutoff)

    if bit_val_in != bit_val_out
        psi_tmp[1] = psi_tmp[1] * op("X", SitesInit[1])
        psi_tmp[1] = noprime(psi_tmp[1])
    end
    if bit_val_in_y != bit_val_out_y
        N::Int = length(psi_tmp)
        psi_tmp[N] = psi_tmp[N] * op("X", SitesInit[N])
        psi_tmp[N] = noprime(psi_tmp[N])
    end

    curr_norm::Float64 = norm(psi_tmp)
    log_gain::Float64  = log(curr_norm)
    normalize!(psi_tmp)
    return psi_tmp, log_gain
end

end
