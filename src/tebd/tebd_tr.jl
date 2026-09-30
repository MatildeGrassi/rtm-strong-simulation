# tebd_tr.jl — transverse contraction of A_xy(T) = ⟨x|U(T)|y⟩ in one left-to-right pass.
# Space and time are swapped: a temporal MPS (2T+2 sites; site 1 = output bit x, site N =
# input bit y) starts at the left spatial edge, absorbs one column (odd + even bond) per step
# with plain SVD truncation to maxbond, and is overlapped with the right edge at the end.
# Writes the final amplitude (one row) to IsingCircuitTEBD/.
# Usage: julia tebd_tr.jl L T gate states iter maxbond [dt hx hz]

using ITensors, ITensorMPS, LinearAlgebra, Random

const FIXED_SEED = 12345
const Id2 = Matrix{ComplexF64}(I, 2, 2)
const Id4 = Matrix{ComplexF64}(I, 4, 4)
const σx = ComplexF64[0 1; 1 0]
const σz = ComplexF64[1 0; 0 -1]

# Kicked-Ising gate: exp(-i[XX + g(XI+IX)/2]) exp(-i h(ZI+IZ)/2), g = 0.81, h = 0.904508.
# dt, J, layer, i, N are not used by the gate itself.
function transverse_field_floquet_ising_gate(dt::Real; J::Real, layer::Int, i::Int, N::Int)
    @assert layer == 1 || layer == 2
    @assert 1 ≤ i ≤ N-1

    XX = kron(σx, σx)
    ZI = kron(σz, Id2)
    IZ = kron(Id2, σz)
    XI = kron(σx, Id2)
    IX = kron(Id2, σx)

    Hbond =  XX +  0.81 * (XI + IX)/2
    Hbond2= 0.904508 * (ZI + IZ)/2

    return ( exp(-1im * Hbond) * exp(-1im * Hbond2))
end

# Trotter gate exp(-i dt H_bond) of the Ising chain, H_bond = J XX + aL H1⊗I + aR I⊗H1,
# H1 = hx X + hz Z. The field is split in half between bonds; an edge site takes it whole.
function twoqubit_floquet_gate_layer(dt::Real; J::Real, hx::Real, hz::Real, layer::Int, i::Int, N::Int, boundary_L::Bool,boundary_R::Bool)
    @assert layer == 1 || layer == 2
    @assert 1 ≤ i ≤ N-1

    H1 = hx*σx + hz*σz
    XX = kron(σx, σx)

    aL::Float64 = boundary_L ? 1.0 : 0.5
    aR::Float64 = boundary_R ? 1.0 : 0.5

    Hbond = J*XX + aL*kron(H1, Id2) + aR*kron(Id2, H1)
    return exp(-1im * dt * Hbond)
end

# Rotated gates of one spatial bond on the temporal chain (bulk, left-edge, right-edge
# versions; they differ only for trotter). Layer 1 → temporal bonds 1, 3, …, 2T+1 (bond 1 is
# the identity), layer 2 → 2, 4, …, 2T. Temporal bonds run from period T (top) down to 1.
function MakeGates(sites, n_sites, dt; J, hx, hz, layer::Int, gate_random::Bool=false, U4_shared_random=nothing,
                   random_gates=nothing, translational_invariance::Bool=true, spatial_bond::Int=1, circuit_steps::Int=1,
                   trotter::Bool=false)
    Gates = ITensor[]
    Gates_boundary_L = ITensor[]
    Gates_boundary_R = ITensor[]

    use_circuit = !trotter

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

# One rotated layer per spatial bond b = 1 … L-1 (odd b → layer 1, even b → layer 2);
# bond 1 and bond L-1 get the left- and right-edge gates.
function make_bond_layers(sites, n_sites, n_sites_fw, steps_fw, dt; J, hx, hz,
                          gate_random, random_gates, translational_invariance, trotter)
    n_bonds  = n_sites_fw - 1
    per_bond = gate_random && !translational_invariance
    layers   = Vector{Vector{ITensor}}(undef, n_bonds)
    for b in 1:n_bonds
        G, G_bL, G_bR = MakeGates(sites, n_sites, dt; J=J, hx=hx, hz=hz,
                                  layer = isodd(b) ? 1 : 2, gate_random=gate_random,
                                  U4_shared_random = per_bond ? nothing : random_gates,
                                  random_gates     = per_bond ? random_gates : nothing,
                                  translational_invariance=translational_invariance,
                                  spatial_bond=b, circuit_steps=steps_fw, trotter=trotter)
        layers[b] = b == 1 ? G_bL : (b == n_bonds ? G_bR : G)
    end
    return layers
end

# 4×4 matrix → rotated gate on temporal sites (i1, i2): columns ↔ (i1, i1'), rows ↔ (i2, i2').
# Unprimed legs = left spatial site, primed legs = right spatial site.
function makeNonUnitaryGate(RandUnitary::Matrix{ComplexF64}, i1::Index, i2::Index)
    i1p = prime(i1)
    i2p = prime(i2)
    G = ITensor(i1, i1p, i2, i2p)
    for row in 1:4
        i2val  = (row-1) ÷ 2 + 1
        i2pval = (row-1) % 2 + 1
        for col in 1:4
            i1val  = (col-1) ÷ 2 + 1
            i1pval = (col-1) % 2 + 1
            G[i1 => i1val, i1p => i1pval,
              i2 => i2val, i2p => i2pval] = RandUnitary[row, col]
        end
    end
    return G
end

# Largest von Neumann entropy over all bonds (weights below eps skipped).
function maxVonNeumannEntropy(psi::MPS)
    N = length(psi)
    working_psi = copy(psi)
    entropies = zeros(N - 1)
    orthogonalize!(working_psi, 1)

    for b in 1:(N - 1)
        # centre moves one site per step, so each SVD sees bond b in canonical form
        orthogonalize!(working_psi, b)
        inds_left = (linkinds(working_psi, b-1)..., siteinds(working_psi, b)...)
        _, S, _ = svd(working_psi[b], inds_left)

        SvN = 0.0
        total = 0.0
        eps = 1.0e-14

        for n in 1:dim(S, 1)
            total += S[n, n]^2
        end

        for n in 1:dim(S, 1)
            p = S[n, n]^2 / total
            if p > eps
                SvN -= p * log(p)
            end
        end

        entropies[b] = SvN
    end
    return maximum(entropies)
end

# Unnormalised Bell creator: |00⟩ → |00⟩ + |11⟩ (basis |00⟩, |01⟩, |10⟩, |11⟩).
function bell_phi_plus_creator_4x4()
    s = 1
    return ComplexF64[
        s  0  0  s;
        0  1  0  0;
        0  0  1  0;
        s  0  0 -s
    ]
end

# 4×4 matrix → ITensor on (i, j): rows = output (primed), columns = input.
function two_site_gate_from_mat(U4::Matrix{ComplexF64}, i::Index, j::Index)
    ip = prime(i); jp = prime(j)
    G = ITensor(i, j, ip, jp)
    for a in 1:2, b in 1:2, ap in 1:2, bp in 1:2
        out = (ap-1)*2 + bp
        inn = (a -1)*2 + b
        G[i=>a, j=>b, ip=>ap, jp=>bp] = U4[out, inn]
    end
    return G
end

# Spatial-edge boundary tMPS: site 1 = x bit, site N = y bit, Bell pairs on (2,3), (4,5), …
# (they join the two legs of each time slice at the open edge).
function product_of_bellpairs_with_zeros_ends_MPS(sites::Vector{<:Index}, first_bitstring, first_bitstring_y)
    N = length(sites)
    @assert iseven(N) "Need even N"
    @assert N ≥ 4 "Need N>=4 to have at least one Bell pair plus fixed ends"

    psi = productMPS(sites, "0")

    if first_bitstring == 1
        psi[1] = psi[1] * op("X", sites[1])
        psi[1] = noprime(psi[1])
    end

    if first_bitstring_y == 1
        psi[N] = psi[N] * op("X", sites[N])
        psi[N] = noprime(psi[N])
    end

    U4 = bell_phi_plus_creator_4x4()
    gates = ITensor[]
    for n in 2:2:N-2
        push!(gates, two_site_gate_from_mat(U4, sites[n], sites[n+1]))
    end

    psi = apply(gates, psi; cutoff=0.0, maxdim=4)

    return psi
end

# Random gates: identical arithmetic and draw order in random_gates.jl (RTM) and tebd_fw.jl.
const SWAP4 = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]

# Haar U0 (QR with phase fix); symmetric = true → polar part of (U0 + SWAP U0 SWAP)/2.
function random_unitary_4x4(rng::AbstractRNG; symmetric::Bool=true)
    Q, R = qr(randn(rng, ComplexF64, 4, 4))
    U0 = Matrix(Q) * Diagonal(sign.(diag(R)))
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

const GATE_TYPES = ("kicked", "sym", "haar", "sym_ti", "haar_ti", "trotter")

# gate word → (gate_random, translational_invariance, haar, trotter, dt, hx, hz).
# `dt hx hz` are required for trotter and rejected otherwise. Same in main.jl, tebd_fw.jl.
function parse_gate(gate::AbstractString, extra::AbstractVector{<:AbstractString})
    gate in GATE_TYPES || error("gate must be one of $(join(GATE_TYPES, ", ")); got \"$(gate)\"")
    trotter = gate == "trotter"
    length(extra) == (trotter ? 3 : 0) ||
        error(trotter ? "gate = trotter needs dt hx hz as the last three arguments" :
                        "dt hx hz are only used with gate = trotter")
    dt, hx, hz = trotter ? parse.(Float64, extra) : (1.0, 0.0, 0.0)
    gate_random = gate in ("sym", "haar", "sym_ti", "haar_ti")
    return gate_random, !(gate in ("sym", "haar")), startswith(gate, "haar"), trotter, dt, hx, hz
end

# Reads the arguments, sweeps the temporal MPS from the left to the right spatial edge and
# writes one row: T, ⟨x|U(T)|y⟩, its modulus, final max entropy, final max bond dimension.
function main()
    n_sites_fw = parse(Int, ARGS[1])
    steps_fw   = parse(Int, ARGS[2])
    gate       = ARGS[3]
    states     = parse(Int, ARGS[4])
    iter       = parse(Int, ARGS[5])
    maxbond    = parse(Int, ARGS[6])
    gate_random, translational_invariance, haar, trotter, dt, hx, hz = parse_gate(gate, ARGS[7:end])
    cutoff= nothing

    # states: 0 → 0→0, 1 → 0→x, 2 → y→x. Independent streams from iter, as in main.jl.
    @assert states in (0, 1, 2) "states must be 0 (0→0), 1 (0→x) or 2 (y→x)"
    rng_x     = MersenneTwister(FIXED_SEED + iter)
    rng_y     = MersenneTwister(UInt32[FIXED_SEED, iter, 2])
    rng_gates = MersenneTwister(UInt32[FIXED_SEED, iter, 3])

    random_gates = gate_random ? random_circuit_gates(rng_gates, n_sites_fw, steps_fw, translational_invariance; symmetric=!haar) : nothing

    bitstring   = states >= 1 ? rand(rng_x, 0:1, n_sites_fw) : zeros(Int, n_sites_fw)
    bitstring_y = states == 2 ? rand(rng_y, 0:1, n_sites_fw) : zeros(Int, n_sites_fw)
    println("x = $(join(bitstring))   y = $(join(bitstring_y))")

    # rotation: L/2 - 1 full columns, temporal chain of 2T + 2 sites
    steps = Int(n_sites_fw/2 - 1)
    n_sites = 2 * steps_fw + 2

    output_dir = "IsingCircuitTEBD"
    isdir(output_dir) || mkdir(output_dir)
    tag = "L$(n_sites_fw)_T$(steps_fw)_gate-$(gate)_states$(states)_iter$(iter)_chi$(maxbond)" *
          (trotter ? "_dt$(dt)_hx$(hx)_hz$(hz)" : "")
    filename = joinpath(output_dir, "Prob.TEBD.TR_$(tag).dat")

    SitesInit = siteinds("Qubit", n_sites)
    total_log_norm = 0.0

    # column s applies spatial bonds 2s-1 (odd) and 2s (even)
    GatesL = make_bond_layers(SitesInit, n_sites, n_sites_fw, steps_fw, dt; J=1.0, hx=hx, hz=hz,
                              gate_random=gate_random, random_gates=random_gates,
                              translational_invariance=translational_invariance, trotter=trotter)

    # left edge (evolved) and right edge (final overlap)
    psi = product_of_bellpairs_with_zeros_ends_MPS(SitesInit,bitstring[1],bitstring_y[1])
    psi0 = product_of_bellpairs_with_zeros_ends_MPS(SitesInit,bitstring[end],bitstring_y[end])

    Sz1 = op("Z", SitesInit[1])
    Id1 = op("I", SitesInit[1])
    SzN = op("Z", SitesInit[n_sites])
    IdN = op("I", SitesInit[n_sites])

    open(filename, "w") do io
        for s in 1:steps
            psi = apply(GatesL[2*s - 1], psi; maxdim=maxbond, cutoff=cutoff)

            # keep the norm aside (log) so the state stays O(1)
            curr_norm = norm(psi)
            total_log_norm += log(curr_norm)
            normalize!(psi)

            # sites 1 / N carry the x / y bit of the current spatial site
            bit_val_out = bitstring[2*s]
            bit_val_in = bitstring[2*s + 1]
            bit_val_out_y = bitstring_y[2*s]
            bit_val_in_y  = bitstring_y[2*s + 1]

            # project sites 1 / N on the bits after the odd layer
            if bit_val_out == 1
                psi[1] = psi[1] * (Id1 - Sz1)/2
            else
                psi[1] = psi[1] * (Id1 + Sz1)/2
            end
            if bit_val_out_y == 1
                psi[n_sites] = psi[n_sites] * (IdN - SzN)/2
            else
                psi[n_sites] = psi[n_sites] * (IdN + SzN)/2
            end
            psi[1] = noprime(psi[1])
            psi[n_sites] = noprime(psi[n_sites])

            psi = apply(GatesL[2*s], psi; maxdim=maxbond, cutoff=cutoff)

            curr_norm = norm(psi)
            total_log_norm += log(curr_norm)
            normalize!(psi)

            # after the even layer: flip sites 1 / N (X) if the next bit differs
            if bit_val_in != bit_val_out
                psi[1] = psi[1] * op("X", SitesInit[1])
                psi[1] = noprime(psi[1])
            end
            if bit_val_in_y != bit_val_out_y
                psi[n_sites] = psi[n_sites] * op("X", SitesInit[n_sites])
                psi[n_sites] = noprime(psi[n_sites])
            end

            # last column: extra odd layer (bond L-1), last bits, overlap with the right edge
            if s == steps
                    psi = apply(GatesL[2*s + 1], psi; maxdim=maxbond, cutoff=cutoff)
                    bit_val_out = bitstring[end]
                    if bit_val_out == 1
                        psi[1] = psi[1] * (Id1 - Sz1)/2
                    else
                        psi[1] = psi[1] * (Id1 + Sz1)/2
                    end
                    psi[1] = noprime(psi[1])
                    bit_val_out_y = bitstring_y[end]
                    if bit_val_out_y == 1
                        psi[n_sites] = psi[n_sites] * (IdN - SzN)/2
                    else
                        psi[n_sites] = psi[n_sites] * (IdN + SzN)/2
                    end
                    psi[n_sites] = noprime(psi[n_sites])

                    curr_norm = norm(psi)
                    total_log_norm += log(curr_norm)
                    normalize!(psi)

                    # physical amplitude = normalised overlap × accumulated norm
                    prob_norm   = abs(inner(psi0, psi))
                    prob_phys = prob_norm * exp(total_log_norm)

                    overlap_norm   = inner(psi0, psi)
                    overlap_phys = overlap_norm * exp(total_log_norm)

                    maxEnt = maxVonNeumannEntropy(psi)

                    println(io, "step    overlap    prob     MaxEnt    Maxbond")
                    println(io,  steps_fw,"    ",overlap_phys,"    ",prob_phys,"    ",maxEnt,"   ",maxlinkdim(psi))
            end
        end
    end
    println("Data saved to: $filename")
end
main()