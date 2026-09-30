# tebd_fw.jl — forward TEBD reference for A_xy(t) = ⟨x|U(t)|y⟩.
# Evolves |y⟩ through T periods (odd then even layer) of a brick-wall circuit and writes
# ⟨x|ψ(t)⟩ after every period to IsingCircuitTEBD/. 
# Usage: julia tebd_fw.jl L T gate states iter maxbond [dt hx hz]

using ITensors, ITensorMPS, LinearAlgebra, Random

const FIXED_SEED = 12345
const Id2 = Matrix{ComplexF64}(I, 2, 2)
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
# H1 = hx X + hz Z. The field is split in half between bonds; edge sites take it whole.
function twoqubit_floquet_gate_layer(dt::Real; J::Real, hx::Real, hz::Real, layer::Int, i::Int, N::Int)
    @assert layer == 1 || layer == 2
    @assert 1 ≤ i ≤ N-1

    H1 = hx*σx + hz*σz
    XX = kron(σx, σx)

    aL::Float64 = (i == 1)   ? 1.0 : 0.5
    aR::Float64 = (i == N-1) ? 1.0 : 0.5
    Hbond = J*XX + aL*kron(H1, Id2) + aR*kron(Id2, H1)
    return exp(-1im * dt * Hbond)
end

# One gate layer: odd bonds (1-2, 3-4, …) if layer = 1, even bonds (2-3, …) if layer = 2.
# Gate: random (shared, or random_gates[(time_step, layer, i)]), kicked, or Trotter.
function MakeGates(sites, n_sites, dt; J, hx, hz, layer::Int,gate_random::Bool=false, U4_shared_random=nothing,
                   random_gates=nothing, translational_invariance::Bool=true,time_step::Int=1, trotter::Bool=false)
    Gates = ITensor[]
    use_circuit = !trotter

    for i in 2-(layer % 2):2:n_sites-1
        if use_circuit
            if gate_random
                if translational_invariance
                    U4=U4_shared_random
                else
                    U4=random_gates[(time_step, layer, i)]
                end
            else
                U4=transverse_field_floquet_ising_gate(dt; J=J, layer=layer, i=i, N=n_sites)
            end
        else
            U4 = twoqubit_floquet_gate_layer(dt; J=J, hx=hx, hz=hz, layer=layer, i=i, N=n_sites)
        end
        push!(Gates, makeUnitaryGate(U4, sites[i], sites[i+1]))
    end
    return Gates
end

# 4×4 matrix → ITensor on sites (i1, i2): rows = output (primed), columns = input.
function makeUnitaryGate(RandUnitary::Matrix{ComplexF64},i1::Index, i2::Index)
    i1p = prime(i1)
    i2p = prime(i2)
    G = ITensor(i1, i2, i1p, i2p)
    for row in 1:4
        i1pval = ((row - 1) ÷ 2) + 1
        i2pval = ((row - 1) % 2) + 1
        for col in 1:4
            i1val = ((col - 1) ÷ 2) + 1
            i2val = ((col - 1) % 2) + 1
            G[i1 => i1val,i2 => i2val,i1p => i1pval,i2p => i2pval] = RandUnitary[row, col]
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

# Random gates: identical arithmetic and draw order in random_gates.jl (RTM) and tebd_tr.jl.
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
# `dt hx hz` are required for trotter and rejected otherwise. Same in main.jl, tebd_tr.jl.
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

# Reads the arguments, evolves |y⟩ and writes one row per period:
# t, ⟨x|ψ(t)⟩, |⟨x|ψ(t)⟩|, max entropy over the bonds, max bond dimension.
function main()
    n_sites = parse(Int, ARGS[1])
    steps   = parse(Int, ARGS[2])
    gate    = ARGS[3]
    states  = parse(Int, ARGS[4])
    iter    = parse(Int, ARGS[5])
    maxbond = parse(Int, ARGS[6])
    gate_random, translational_invariance, haar, trotter, dt, hx, hz = parse_gate(gate, ARGS[7:end])
    cutoff = nothing

    # states: 0 → 0→0, 1 → 0→x, 2 → y→x. Independent streams from iter, as in main.jl.
    @assert states in (0, 1, 2) "states must be 0 (0→0), 1 (0→x) or 2 (y→x)"
    rng_x     = MersenneTwister(FIXED_SEED + iter)
    rng_y     = MersenneTwister(UInt32[FIXED_SEED, iter, 2])
    rng_gates = MersenneTwister(UInt32[FIXED_SEED, iter, 3])

    random_gates = gate_random ? random_circuit_gates(rng_gates, n_sites, steps, translational_invariance; symmetric=!haar) : nothing

    bitstring   = states >= 1 ? rand(rng_x, 0:1, n_sites) : zeros(Int, n_sites)
    bitstring_y = states == 2 ? rand(rng_y, 0:1, n_sites) : zeros(Int, n_sites)
    println("x = $(join(bitstring))   y = $(join(bitstring_y))")

    output_dir = "IsingCircuitTEBD"
    isdir(output_dir) || mkdir(output_dir)
    tag = "L$(n_sites)_T$(steps)_gate-$(gate)_states$(states)_iter$(iter)_chi$(maxbond)" *
          (trotter ? "_dt$(dt)_hx$(hx)_hz$(hz)" : "")
    filename = joinpath(output_dir, "Prob.TEBD.FW_$(tag).dat")

    # psi0 = |x⟩ (output), psi = |y⟩ (input, evolved)
    sites = siteinds("Qubit", n_sites)
    init_state = [bitstring[i] == 1 ? "1" : "0" for i in 1:n_sites]
    psi0 = productMPS(sites, init_state)
    input_state = [bitstring_y[i] == 1 ? "1" : "0" for i in 1:n_sites]
    psi = productMPS(sites, input_state)

    # Layers built once, unless the gates change with the period (sym, haar)
    GatesOdd  = translational_invariance || !gate_random ? MakeGates(sites, n_sites, dt; J=1.0, hx=hx, hz=hz, layer=1, gate_random, U4_shared_random=random_gates, trotter=trotter) : nothing
    GatesEven = translational_invariance || !gate_random ? MakeGates(sites, n_sites, dt; J=1.0, hx=hx, hz=hz, layer=2, gate_random, U4_shared_random=random_gates, trotter=trotter) : nothing

    open(filename, "w") do io
        println(io, "step    overlap    prob    MaxEnt    Maxbond")
        for s in 1:steps
            gates_odd = translational_invariance || !gate_random ? GatesOdd : MakeGates(sites, n_sites, dt; J=1.0, hx=hx, hz=hz, layer=1, gate_random, random_gates=random_gates, translational_invariance=false, time_step=s)
            gates_even = translational_invariance || !gate_random ? GatesEven : MakeGates(sites, n_sites, dt; J=1.0, hx=hx, hz=hz, layer=2, gate_random, random_gates=random_gates, translational_invariance=false, time_step=s)

            # one period: odd then even layer; renormalise (only truncation changes the norm)
            psi = apply(gates_odd, psi; maxdim=maxbond, cutoff=cutoff)
            normalize!(psi)

            psi = apply(gates_even, psi; maxdim=maxbond, cutoff=cutoff)
            normalize!(psi)

            maxEnt = maxVonNeumannEntropy(psi)

            prob   = abs(inner(psi0, psi))

            overlap   = inner(psi0, psi)

            println(io,  s,"    ",overlap,"    ",prob,"    ",maxEnt,"    ",maxlinkdim(psi))
        end
    end
    println("Data saved to: $filename")
end
main()