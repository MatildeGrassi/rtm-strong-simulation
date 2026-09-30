# main.jl — RTM strong simulation of A_xy(T) = ⟨x|U(T)|y⟩ by transverse contraction.
# Space and time are swapped: L/2 columns, each a temporal MPS of 2T+2 sites (site 1 = x bit,
# site N = y bit). Builds the boundary MPS, then runs the RTM sweeps (rtm.jl), which write
# the per-cut amplitudes to IsingCircuitRTM/.
# Usage: julia main.jl L T gate states iter maxbond conv_threshold nstart [dt hx hz]
#   gate    kicked | sym | haar | sym_ti | haar_ti | trotter (trotter needs dt hx hz)
#   states  0: 0→0, 1: 0→x, 2: y→x (x, y random, fixed by iter)
#   maxbond bond cap of the gate layers in the sweeps (not the RTM χ)
#   conv_threshold  tolerance of the sweep diagnostics; nstart  first n of χ = 16 + n²

include("random_gates.jl");   using .RandomGates
include("gates.jl");          using .Gates
include("entanglement.jl")
include("initial_states.jl"); using .InitialStates
include("rtm.jl");            using .RTM

using ITensors, ITensorMPS, Random

const FIXED_SEED::Int = 12345
# bond cap of the gate layers that build the initial psiR
const STARTBOND_RDM::Int = 10
const GATE_TYPES = ("kicked", "sym", "haar", "sym_ti", "haar_ti", "trotter")

# gate word → (gate_random, translational_invariance, haar, trotter, dt, hx, hz).
# `dt hx hz` are required for trotter and rejected otherwise. Same in tebd_fw.jl, tebd_tr.jl.
function parse_gate(gate::AbstractString, extra::AbstractVector{<:AbstractString})
    gate in GATE_TYPES || error("gate must be one of $(join(GATE_TYPES, ", ")); got \"$(gate)\"")
    trotter::Bool = gate == "trotter"
    length(extra) == (trotter ? 3 : 0) ||
        error(trotter ? "gate = trotter needs dt hx hz as the last three arguments" :
                        "dt hx hz are only used with gate = trotter")
    dt, hx, hz = trotter ? parse.(Float64, extra) : (1.0, 0.0, 0.0)
    gate_random::Bool = gate in ("sym", "haar", "sym_ti", "haar_ti")
    return gate_random, !(gate in ("sym", "haar")), startswith(gate, "haar"), trotter, dt, hx, hz
end

# Independent random streams for x, y and the gates, all fixed by iter (same in the TEBD codes).
function make_rngs(iter::Int)::NTuple{3, MersenneTwister}
    rng_x::MersenneTwister     = MersenneTwister(FIXED_SEED + iter)
    rng_y::MersenneTwister     = MersenneTwister(UInt32[FIXED_SEED, iter, 2])
    rng_gates::MersenneTwister = MersenneTwister(UInt32[FIXED_SEED, iter, 3])
    return rng_x, rng_y, rng_gates
end

# Random bitstring of length L, or all zeros.
function generate_bitstring(n_sites_fw::Int, random::Bool, rng::AbstractRNG)::Vector{Int}
    return random ? rand(rng, 0:1, n_sites_fw) : zeros(Int, n_sites_fw)
end

# Initial right boundaries psiR[s] (and their log-norms): start from the right-edge Bell
# state and move left one column at a time. psiR[s] → psiR[s-1] applies bonds 2s-2 (even)
# and 2s-3 (odd); the rightmost column first applies bond L-1 as well.
# NB: the bulk bits here are shifted by one w.r.t. the backward sweep in rtm.jl; this only
# affects the starting guess (psiR is rebuilt by the first backward sweep).
function build_psiR(SitesInit::Vector{<:Index},
                    n_sites::Int,
                    steps::Int,
                    bitstring::Vector{Int},
                    bitstring_y::Vector{Int},
                    GatesR::Vector{Vector{ITensor}},
                    maxbond::Int)::Tuple{Vector{MPS}, Vector{Float64}}
    cutoff::Float64 = 0.0
    Id1::ITensor = op("I", SitesInit[1])
    Sz1::ITensor = op("Z", SitesInit[1])
    IdN::ITensor = op("I", SitesInit[n_sites])
    SzN::ITensor = op("Z", SitesInit[n_sites])

    psiR::Vector{MPS}       = Vector{MPS}(undef, steps)
    psiR[steps]             = product_of_bellpairs_with_zeros_ends_MPS(SitesInit, bitstring[end],
                                                                       bitstring_y[end])
    log_norm_R::Vector{Float64} = zeros(Float64, steps)

    for s in steps:-1:2
        psi_tmp::MPS          = copy(psiR[s])
        log_norm_tmp::Float64 = log_norm_R[s]

        if s == steps
            # rightmost column: odd (L-1), even (L-2), odd (L-3)
            bit_val_out::Int   = bitstring[end-1]
            bit_val_in::Int    = bitstring[end-2]
            bit_val_out_y::Int = bitstring_y[end-1]
            bit_val_in_y::Int  = bitstring_y[end-2]

            psi_tmp, curr_norm = apply_odd_layer!(psi_tmp, GatesR[2*steps - 1], maxbond, cutoff,
                                                  bit_val_out, bit_val_out_y, Id1, Sz1, IdN, SzN)
            log_norm_tmp += curr_norm

            psi_tmp, curr_norm = apply_even_layer!(psi_tmp, GatesR[2*steps - 2], maxbond, cutoff,
                                                   bit_val_in, bit_val_out,
                                                   bit_val_in_y, bit_val_out_y, SitesInit)
            log_norm_tmp += curr_norm

            bit_val_out   = bitstring[end-3]
            bit_val_out_y = bitstring_y[end-3]
            psi_tmp, curr_norm = apply_odd_layer!(psi_tmp, GatesR[2*steps - 3], maxbond, cutoff,
                                                  bit_val_out, bit_val_out_y, Id1, Sz1, IdN, SzN)
            log_norm_tmp += curr_norm
        else
            bit_val_out   = bitstring[2*(s-1)]
            bit_val_in    = bitstring[2*(s-1)-1]
            bit_val_out_y = bitstring_y[2*(s-1)]
            bit_val_in_y  = bitstring_y[2*(s-1)-1]

            psi_tmp, curr_norm = apply_even_layer!(psi_tmp, GatesR[2*s - 2], maxbond, cutoff,
                                                   bit_val_in, bit_val_out,
                                                   bit_val_in_y, bit_val_out_y, SitesInit)
            log_norm_tmp += curr_norm

            bit_val_out   = bitstring[2*(s-1)+1]
            bit_val_out_y = bitstring_y[2*(s-1)+1]
            psi_tmp, curr_norm = apply_odd_layer!(psi_tmp, GatesR[2*s - 3],
                                                  maxbond, cutoff, bit_val_out, bit_val_out_y,
                                                  Id1, Sz1, IdN, SzN)
            log_norm_tmp += curr_norm
        end

        log_norm_R[s-1] = log_norm_tmp
        psiR[s-1]       = psi_tmp
    end
    return psiR, log_norm_R
end

# Reads the arguments, builds gates and boundary MPS, and runs the RTM sweeps.
function main()
    n_sites_fw::Int         = parse(Int,     ARGS[1])
    steps_fw::Int           = parse(Int,     ARGS[2])
    gate::String            = ARGS[3]
    states::Int             = parse(Int,     ARGS[4])
    iter::Int               = parse(Int,     ARGS[5])
    maxbond::Int            = parse(Int,     ARGS[6])
    conv_threshold::Float64 = parse(Float64, ARGS[7])
    nstart::Int             = parse(Int,     ARGS[8])
    gate_random, translational_invariance, haar, trotter, dt, hx, hz = parse_gate(gate, ARGS[9:end])
    println("RTM run parameters: L=$(n_sites_fw) T=$(steps_fw) gate=$(gate) states=$(states) " *
            "iter=$(iter) maxbond=$(maxbond) conv_threshold=$(conv_threshold) nstart=$(nstart)" *
            (trotter ? " dt=$(dt) hx=$(hx) hz=$(hz)" : ""))

    @assert states in (0, 1, 2) "states must be 0 (0→0), 1 (0→x) or 2 (y→x)"
    rng_x, rng_y, rng_gates = make_rngs(iter)

    bitstring::Vector{Int}   = generate_bitstring(n_sites_fw, states >= 1, rng_x)
    bitstring_y::Vector{Int} = generate_bitstring(n_sites_fw, states == 2, rng_y)
    println("x = $(join(bitstring))   y = $(join(bitstring_y))")

    random_gates = gate_random ? random_circuit_gates(rng_gates, n_sites_fw, steps_fw,
                                                       translational_invariance;
                                                       symmetric=!haar) : nothing

    # rotation: L/2 columns (cuts), temporal chain of 2T + 2 sites
    steps::Int   = Int(n_sites_fw / 2)
    n_sites::Int = 2 * steps_fw + 2

    SitesInit::Vector{<:Index} = siteinds("Qubit", n_sites)

    # one rotated layer per spatial bond: GatesL evolves psiL, GatesR evolves psiR
    GatesL, GatesR = make_bond_layers(SitesInit, n_sites, n_sites_fw, steps_fw, dt;
                                      J=1.0, hx=hx, hz=hz, gate_random=gate_random,
                                      random_gates=random_gates,
                                      translational_invariance=translational_invariance,
                                      trotter=trotter)

    # psiL[1] = left-edge Bell state; the other psiL are built by the first forward sweep
    psiL::Vector{MPS}         = Vector{MPS}(undef, steps)
    psiL[1]                   = product_of_bellpairs_with_zeros_ends_MPS(SitesInit, bitstring[1],
                                                                         bitstring_y[1])
    log_norm_L::Vector{Float64} = zeros(Float64, steps)

    psiR::Vector{MPS}, log_norm_R::Vector{Float64} =
        build_psiR(SitesInit, n_sites, steps, bitstring, bitstring_y,
                   GatesR, STARTBOND_RDM)

    output_dir::String = "IsingCircuitRTM"
    isdir(output_dir) || mkdir(output_dir)
    tag::String = "L$(n_sites_fw)_T$(steps_fw)_gate-$(gate)_states$(states)_iter$(iter)_chi$(maxbond)" *
                  (trotter ? "_dt$(dt)_hx$(hx)_hz$(hz)" : "") * "_eps$(conv_threshold)"
    filename::String    = joinpath(output_dir, "Prob.RTM_$(tag).dat")
    sv_filename::String = joinpath(output_dir, "rtm_singular_values_$(tag).dat")

    open(filename, "w") do io
        println(io, "# conv_threshold = $(conv_threshold)")
        rtm_sweeps!(psiL, psiR, log_norm_L, log_norm_R, bitstring, bitstring_y,
                            GatesL, GatesR, SitesInit, n_sites, steps,
                            maxbond, io, sv_filename, n_sites_fw, steps_fw;
                            conv_threshold=conv_threshold, nstart=nstart)
    end
end

main()
