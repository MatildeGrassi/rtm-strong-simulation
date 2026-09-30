# rtm.jl — RTM truncation and the adaptive sweeps of the transverse contraction.
# psiL[k] / psiR[k] are the temporal MPS left / right of cut k; A_k = ⟨L_k|R_k⟩ (times the
# stored norms) is the amplitude read at cut k. χ = 16 + n², one block of L/2 sweeps per n.
# At a block end the run becomes a candidate if Δcut, Δsweep < ε; one more block at the next
# χ must then give Δcut, Δsweep, Δχ < ε to stop (Eq. (10) of the paper).

module RTM

using ITensors, ITensorMPS, LinearAlgebra, Printf

# sibling modules are loaded by main.jl (no include here, to avoid duplicate modules)
using ..Gates
using ..Entanglement: entropy_from_singular_values,
                      sumsq_singular_values, renyi2_from_singular_values

include("linear_svd.jl")

export rtm_sweeps!

const RTM_MAXBOND_BASE::Int    = 16
const RTM_MAXBOND_COEFF::Int   = 1
const RTM_MAXBOND_CEILING::Int = 1024
# stop at sweep 1 if the first N_OVERLAP_CHECK cuts all give |A| < OVERLAP_MIN
const OVERLAP_MIN::Float64   = 1.0e-12
const N_OVERLAP_CHECK::Int   = 5

# One RTM truncation of the pair (psiL, psiR) to bond dimension ≤ rtm_maxbond.
# From the last temporal site down to site 2: grow the environment E = L·R, split it by SVD
# (largest rtm_maxbond values) and absorb U, V into both MPS, so what is kept is what matters
# for ⟨L|R⟩. Returns overlap, psiL, psiR, the largest spectrum entropy, the spectrum of that
# bond (first one on ties) and the spectrum of the middle temporal bond.
function rtm_contraction(psiL_in::MPS, psiR_in::MPS;
                          rtm_maxbond::Int)::Tuple{ComplexF64, MPS, MPS, Float64, Vector{Float64}, Vector{Float64}}
    psiL::MPS = copy(psiL_in)
    psiR::MPS = copy(psiR_in)

    Nt::Int             = length(psiL)
    max_entropy_rtm::Float64           = -Inf
    max_entropy_svals::Vector{Float64} = Float64[]

    # middle temporal bond (b_mid, b_mid+1); the SVD at site j cuts bond j-1
    b_mid::Int                     = fld(Nt, 2)
    mid_svals::Vector{Float64}     = Float64[]

    orthogonalize!(psiL, Nt)
    orthogonalize!(psiR, Nt)

    # E carries psiL's link first, which is where svd_linear splits it
    E::ITensor = psiL[Nt] * psiR[Nt]

    U, S, V = svd_linear(E; maxdim=rtm_maxbond)
    entropy_here::Float64 = entropy_from_singular_values(S)
    if entropy_here > max_entropy_rtm
        max_entropy_rtm   = entropy_here
        max_entropy_svals = Float64[abs(S[n, n]) for n in 1:dim(S, 1)]
    end
    (Nt - 1) == b_mid && (mid_svals = Float64[abs(S[n, n]) for n in 1:dim(S, 1)])

    # U†U and V†V inserted across the bond: projectors onto the kept directions
    psiL[Nt]     = psiL[Nt]     * dag(U)
    psiR[Nt]     = psiR[Nt]     * dag(V)
    psiL[Nt - 1] = psiL[Nt - 1] * U
    psiR[Nt - 1] = psiR[Nt - 1] * V
    E *= dag(U)
    E *= dag(V)

    for j in (Nt - 1):-1:2
        Lj::ITensor = psiL[j]
        Rj::ITensor = psiR[j]

        E *= Lj
        E *= Rj

        U, S, V = svd_linear(E; maxdim=rtm_maxbond)
        entropy_here = entropy_from_singular_values(S)
        if entropy_here > max_entropy_rtm
            max_entropy_rtm   = entropy_here
            max_entropy_svals = Float64[abs(S[n, n]) for n in 1:dim(S, 1)]
        end
        (j - 1) == b_mid && (mid_svals = Float64[abs(S[n, n]) for n in 1:dim(S, 1)])

        psiL[j]     = Lj          * dag(U)
        psiR[j]     = Rj          * dag(V)
        psiL[j - 1] = psiL[j - 1] * U
        psiR[j - 1] = psiR[j - 1] * V
        E *= dag(U)
        E *= dag(V)
    end

    overlap::ComplexF64 = ComplexF64(inner(dag(psiL), psiR))
    return overlap, psiL, psiR, max_entropy_rtm, max_entropy_svals, mid_svals
end

# Symmetric relative difference |a − b| / |(a + b)/2| (0 if equal, Inf if the midpoint is 0).
function rel_diff(a::Number, b::Number)::Float64
    num = abs(a - b)
    num == 0 && return 0.0
    den = abs(a + b) / 2
    den == 0 && return Inf
    return num / den
end

# Largest rel_diff over all pairs (A[j], B[k]); Inf if a vector is empty, NaN propagates.
function max_rel_diff(A::AbstractVector{<:Number}, B::AbstractVector{<:Number})::Float64
    (isempty(A) || isempty(B)) && return Inf
    worst = 0.0
    for a in A, b in B
        worst = max(worst, rel_diff(a, b))
    end
    return worst
end

# Diagnostics of Eq. (10): all pairs of cuts within one sweep (Δcut), between consecutive
# sweeps (Δsweep), and between the check block and the candidate (Δχ).
cut_spread(overlaps)                  = max_rel_diff(overlaps, overlaps)
sweep_change(overlaps, prev_overlaps) = max_rel_diff(prev_overlaps, overlaps)
chi_change(overlaps, cand_overlaps)   = max_rel_diff(overlaps, cand_overlaps)

# Adaptive RTM sweeps. Each sweep: forward pass (update psiL[2..steps]) and backward pass
# (update psiR[steps-1..1]), each step absorbing one column and truncating with rtm_contraction.
# Writes one row per cut and sweep to io, and the last sweep's spectra to sv_filename.
# maxbond caps the gate-layer applications; the RTM χ follows the block schedule.
function rtm_sweeps!(psiL_n::Vector{MPS},
                     psiR_n::Vector{MPS},
                     log_norm_L_n::Vector{Float64},
                     log_norm_R_n::Vector{Float64},
                     bitstring::Vector{Int},
                     bitstring_y::Vector{Int},
                     GatesL::Vector{Vector{ITensor}},
                     GatesR::Vector{Vector{ITensor}},
                     SitesInit::Vector{<:Index},
                     n_sites::Int,
                     steps::Int,
                     maxbond::Int,
                     io::IO,
                     sv_filename::String,
                     n_sites_fw::Int,
                     steps_fw::Int;
                     conv_threshold::Float64,
                     nstart::Int = 1)
    Id1::ITensor = op("I", SitesInit[1])
    Sz1::ITensor = op("Z", SitesInit[1])
    IdN::ITensor = op("I", SitesInit[n_sites])
    SzN::ITensor = op("Z", SitesInit[n_sites])

    const_no_cutoff::Nothing = nothing

    # sweeps per block; last block n whose χ stays below the ceiling (n = 31 → χ = 977)
    rtm_maxbond_period::Int = div(n_sites_fw, 2)
    n_max_block::Int = isqrt(fld(RTM_MAXBOND_CEILING - RTM_MAXBOND_BASE, RTM_MAXBOND_COEFF))
    if nstart > n_max_block
        error("nstart = $(nstart) already exceeds the bond ceiling " *
              "($(RTM_MAXBOND_CEILING)): the first block's rtm_maxbond would be " *
              "$(RTM_MAXBOND_BASE + RTM_MAXBOND_COEFF * nstart^2). " *
              "Choose nstart ≤ $(n_max_block).")
    end
    max_sweeps_safety::Int = (n_max_block - nstart + 1) * rtm_maxbond_period
    rtm_maxbond::Int                  = 0
    sweep::Int                        = 0
    prev_overlaps::Vector{ComplexF64} = ComplexF64[]
    ref_cut::Int                      = cld(steps, 2)
    # true while the check block is running
    converged_flag::Bool              = false
    candidate_overlaps::Vector{ComplexF64} = ComplexF64[]
    candidate_maxbond::Int                 = 0
    # spectra of the current sweep: max-entropy bond and middle bond of each cut
    sv_spectra::Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, steps)
    sv_spectra_mid::Vector{Vector{Float64}} = Vector{Vector{Float64}}(undef, steps)

    while true
        sweep += 1
        # χ = 16 + n², n = nstart + (block index)
        n_step::Int = nstart + cld(sweep, rtm_maxbond_period) - 1
        rtm_maxbond = min(RTM_MAXBOND_BASE + RTM_MAXBOND_COEFF * n_step^2,
                          RTM_MAXBOND_CEILING)
        max_sv_entropy::Vector{Float64} = zeros(Float64, steps)
        sv_spectra = [Float64[] for _ in 1:steps]
        sv_spectra_mid = [Float64[] for _ in 1:steps]
        println("SWEEP $(sweep)  (n = $(n_step), rtm_maxbond = $(rtm_maxbond))")

        # forward pass: psiL[s] → psiL[s+1] (bonds 2s-1, 2s), truncated against the
        # partner psiR[s+2] moved one column left (bonds 2s+2, 2s+1)
        for s in 1:(steps-2)
            psi_tmp_L::MPS       = copy(psiL_n[s])
            log_norm_tmp_L::Float64 = log_norm_L_n[s]
            psi_tmp_R::MPS       = copy(psiR_n[s + 2])
            log_norm_tmp_R::Float64 = log_norm_R_n[s + 2]

            bit_val_out::Int   = bitstring[2*s]
            bit_val_in::Int    = bitstring[2*s + 1]
            bit_val_out_y::Int = bitstring_y[2*s]
            bit_val_in_y::Int  = bitstring_y[2*s + 1]
            psi_tmp_L, curr_norm = apply_odd_layer!(psi_tmp_L, GatesL[2*s - 1], maxbond,
                                                    const_no_cutoff, bit_val_out, bit_val_out_y,
                                                    Id1, Sz1, IdN, SzN)
            log_norm_tmp_L += curr_norm

            psi_tmp_L, curr_norm = apply_even_layer!(psi_tmp_L, GatesL[2*s], maxbond,
                                                     const_no_cutoff, bit_val_in, bit_val_out,
                                                     bit_val_in_y, bit_val_out_y, SitesInit)
            log_norm_tmp_L += curr_norm

            # rightmost column also carries bond L-1
            if s == (steps - 2)
                bit_val_out   = bitstring[end-1]
                bit_val_out_y = bitstring_y[end-1]
                psi_tmp_R, curr_norm = apply_odd_layer!(psi_tmp_R, GatesR[2*steps - 1], maxbond,
                                                        const_no_cutoff, bit_val_out, bit_val_out_y,
                                                        Id1, Sz1, IdN, SzN)
                log_norm_tmp_R += curr_norm
            end

            bit_val_in    = bitstring[2*s + 2]
            bit_val_out   = bitstring[2*s + 3]
            bit_val_in_y  = bitstring_y[2*s + 2]
            bit_val_out_y = bitstring_y[2*s + 3]
            psi_tmp_R, curr_norm = apply_even_layer!(psi_tmp_R, GatesR[2*s + 2], maxbond,
                                                     const_no_cutoff, bit_val_in, bit_val_out,
                                                     bit_val_in_y, bit_val_out_y, SitesInit)
            log_norm_tmp_R += curr_norm

            bit_val_out   = bitstring[2*s + 1]
            bit_val_out_y = bitstring_y[2*s + 1]
            psi_tmp_R, curr_norm = apply_odd_layer!(psi_tmp_R, GatesR[2*s + 1], maxbond,
                                                        const_no_cutoff, bit_val_out, bit_val_out_y,
                                                        Id1, Sz1, IdN, SzN)
            log_norm_tmp_R += curr_norm

            # only psiL is kept; psiR[s+1] is not overwritten here
            _, psiL_new::MPS, _, _ = rtm_contraction(psi_tmp_L, psi_tmp_R; rtm_maxbond)

            log_norm_L_n[s+1] = log_norm_tmp_L
            log_norm_R_n[s+1] = log_norm_tmp_R
            psiL_n[s+1]       = psiL_new
        end

        # last forward step: psiL[steps-1] → psiL[steps] (bonds L-3, L-2, L-1),
        # truncated against the right-edge state psiR[steps]
        psi_tmp_L       = copy(psiL_n[steps-1])
        log_norm_tmp_L  = log_norm_L_n[steps-1]

        bit_val_out   = bitstring[2*(steps-1)]
        bit_val_in    = bitstring[2*(steps-1) + 1]
        bit_val_out_y = bitstring_y[2*(steps-1)]
        bit_val_in_y  = bitstring_y[2*(steps-1) + 1]

        psi_tmp_L, curr_norm = apply_odd_layer!(psi_tmp_L, GatesL[2*steps - 3], maxbond,
                                                const_no_cutoff, bit_val_out, bit_val_out_y,
                                                Id1, Sz1, IdN, SzN)
        log_norm_tmp_L += curr_norm

        psi_tmp_L, curr_norm = apply_even_layer!(psi_tmp_L, GatesL[2*steps - 2], maxbond,
                                                 const_no_cutoff, bit_val_in, bit_val_out,
                                                 bit_val_in_y, bit_val_out_y, SitesInit)
        log_norm_tmp_L += curr_norm

        bit_val_out   = bitstring[end]
        bit_val_out_y = bitstring_y[end]
        psi_tmp_L, curr_norm = apply_odd_layer!(psi_tmp_L, GatesL[2*steps - 1], maxbond,
                                                const_no_cutoff, bit_val_out, bit_val_out_y,
                                                Id1, Sz1, IdN, SzN)
        log_norm_tmp_L += curr_norm

        _, psiL_new, _, _ = rtm_contraction(psi_tmp_L, psiR_n[steps]; rtm_maxbond)

        log_norm_L_n[steps] = log_norm_tmp_L
        psiL_n[steps]       = psiL_new
        GC.gc()

        # backward pass: psiR[s] → psiR[s-1] (bonds 2s-2, 2s-3; also L-1 when s = steps),
        # truncated against psiL[s-2] moved one column right (bonds 2s-5, 2s-4)
        for s in steps:-1:3
            psi_tmp_R       = copy(psiR_n[s])
            log_norm_tmp_R  = log_norm_R_n[s]
            psi_tmp_L       = copy(psiL_n[s-2])
            log_norm_tmp_L  = log_norm_L_n[s-2]

            if s == steps
                bit_val_out   = bitstring[end-1]
                bit_val_out_y = bitstring_y[end-1]
                psi_tmp_R, curr_norm = apply_odd_layer!(psi_tmp_R, GatesR[2*steps - 1], maxbond,
                                                        const_no_cutoff, bit_val_out, bit_val_out_y,
                                                        Id1, Sz1, IdN, SzN)
                log_norm_tmp_R += curr_norm

                bit_val_in   = bitstring[end-2]
                bit_val_in_y = bitstring_y[end-2]
                psi_tmp_R, curr_norm = apply_even_layer!(psi_tmp_R, GatesR[2*s - 2], maxbond,
                                                         const_no_cutoff, bit_val_in, bit_val_out,
                                                         bit_val_in_y, bit_val_out_y, SitesInit)
                log_norm_tmp_R += curr_norm

                bit_val_out   = bitstring[end-3]
                bit_val_out_y = bitstring_y[end-3]
                psi_tmp_R, curr_norm = apply_odd_layer!(psi_tmp_R, GatesR[2*s - 3], maxbond,
                                                        const_no_cutoff, bit_val_out, bit_val_out_y,
                                                        Id1, Sz1, IdN, SzN)
                log_norm_tmp_R += curr_norm
            else
                bit_val_out   = bitstring[2*s - 1]
                bit_val_in    = bitstring[2*s - 2]
                bit_val_out_y = bitstring_y[2*s - 1]
                bit_val_in_y  = bitstring_y[2*s - 2]

                psi_tmp_R, curr_norm = apply_even_layer!(psi_tmp_R, GatesR[2*s - 2], maxbond,
                                                         const_no_cutoff, bit_val_in, bit_val_out,
                                                         bit_val_in_y, bit_val_out_y, SitesInit)
                log_norm_tmp_R += curr_norm

                bit_val_out   = bitstring[2*s - 3]
                bit_val_out_y = bitstring_y[2*s - 3]
                psi_tmp_R, curr_norm = apply_odd_layer!(psi_tmp_R, GatesR[2*s - 3], maxbond,
                                                        const_no_cutoff, bit_val_out, bit_val_out_y,
                                                        Id1, Sz1, IdN, SzN)
                log_norm_tmp_R += curr_norm
            end

            bit_val_out   = bitstring[2*(s-2)]
            bit_val_in    = bitstring[2*(s-2) + 1]
            bit_val_out_y = bitstring_y[2*(s-2)]
            bit_val_in_y  = bitstring_y[2*(s-2) + 1]
            psi_tmp_L, curr_norm = apply_odd_layer!(psi_tmp_L, GatesL[2*s - 5], maxbond,
                                                    const_no_cutoff, bit_val_out, bit_val_out_y,
                                                    Id1, Sz1, IdN, SzN)
            log_norm_tmp_L += curr_norm

            psi_tmp_L, curr_norm = apply_even_layer!(psi_tmp_L, GatesL[2*s - 4], maxbond,
                                                     const_no_cutoff, bit_val_in, bit_val_out,
                                                     bit_val_in_y, bit_val_out_y, SitesInit)
            log_norm_tmp_L += curr_norm

            _, _, psiR_new::MPS, max_sv_entropy[s-1], sv_spectra[s-1], sv_spectra_mid[s-1] =
                rtm_contraction(psi_tmp_L, psi_tmp_R; rtm_maxbond)

            log_norm_L_n[s-1] = log_norm_tmp_L
            log_norm_R_n[s-1] = log_norm_tmp_R
            psiR_n[s-1]       = psiR_new
        end

        # last backward step: psiR[2] → psiR[1] (bonds 2, 1), against the left edge psiL[1]
        psi_tmp_R      = copy(psiR_n[2])
        log_norm_tmp_R = log_norm_R_n[2]

        bit_val_out   = bitstring[3]
        bit_val_in    = bitstring[2]
        bit_val_out_y = bitstring_y[3]
        bit_val_in_y  = bitstring_y[2]

        psi_tmp_R, curr_norm = apply_even_layer!(psi_tmp_R, GatesR[2], maxbond,
                                                 const_no_cutoff, bit_val_in, bit_val_out,
                                                 bit_val_in_y, bit_val_out_y, SitesInit)
        log_norm_tmp_R += curr_norm

        bit_val_out   = bitstring[1]
        bit_val_out_y = bitstring_y[1]
        psi_tmp_R, curr_norm = apply_odd_layer!(psi_tmp_R, GatesR[1], maxbond,
                                                const_no_cutoff, bit_val_out, bit_val_out_y,
                                                Id1, Sz1, IdN, SzN)
        log_norm_tmp_R += curr_norm

        _, _, psiR_new, max_sv_entropy[1], sv_spectra[1], sv_spectra_mid[1] =
            rtm_contraction(psiL_n[1], psi_tmp_R; rtm_maxbond)

        log_norm_R_n[1] = log_norm_tmp_R
        psiR_n[1]       = psiR_new
        GC.gc()

        # spectra of the last cut (the backward pass does not visit it)
        _, _, _, max_sv_entropy[end], sv_spectra[end], sv_spectra_mid[end] =
            rtm_contraction(psiL_n[end], psiR_n[end]; rtm_maxbond)

        # amplitude at every cut: normalised overlap × stored norms
        overlaps::Vector{ComplexF64} = zeros(ComplexF64, steps)
        probs::Vector{Float64}       = zeros(Float64, steps)
        for k in 1:steps
            overlap_k::ComplexF64 = ComplexF64(inner(dag(psiR_n[k]), psiL_n[k])) *
                                    exp(log_norm_R_n[k] + log_norm_L_n[k])
            overlaps[k] = overlap_k
            probs[k]    = abs(overlap_k)
        end

        if sweep == 1
            ncheck::Int = min(N_OVERLAP_CHECK, steps)
            if all(k -> !isfinite(probs[k]) || probs[k] < OVERLAP_MIN, 1:ncheck)
                msg = "ABORT: the first $(ncheck) overlaps of sweep 1 are below " *
                      "$(OVERLAP_MIN) (max |⟨L|R⟩| = $(maximum(probs[1:ncheck]))); " *
                      "bitstring = $(join(bitstring)). Stopping the run."
                println(msg)
                println(io, "# $(msg)")
                flush(io)
                error(msg)
            end
        end

        spread::Float64 = cut_spread(overlaps)
        change::Float64 = sweep == 1 ? NaN : sweep_change(overlaps, prev_overlaps)

        # row: sweep, k/steps, Re A, Im A, |A|, Δcut, Δsweep, maxlinkdim L, R,
        # max spectrum entropy, Rényi-2 of the middle temporal bond
        for k in 1:steps
            mid_renyi2::Float64 = renyi2_from_singular_values(sv_spectra_mid[k])
            println(io, "$sweep $k/$steps $(real(overlaps[k])) $(imag(overlaps[k])) " *
                        "$(probs[k]) $spread $change " *
                        "$(maxlinkdim(psiL_n[k])) $(maxlinkdim(psiR_n[k])) " *
                        "$(max_sv_entropy[k]) $(mid_renyi2)")
        end

        if sweep == 1
            println("  CUT SPREAD = $(spread)")
        else
            println("  CUT SPREAD = $(spread)   SWEEP CHANGE = $(change)")
        end

        # decisions only at block ends; a failed check block is not a new candidate
        is_block_end::Bool = sweep % rtm_maxbond_period == 0
        criteria_met::Bool = spread < conv_threshold && change < conv_threshold

        if converged_flag
            if !criteria_met
                println("  WARNING: sweep $(sweep) (check block) does not satisfy " *
                        "the convergence criteria (CUT SPREAD = $(spread), " *
                        "SWEEP CHANGE = $(change), threshold = $(conv_threshold)).")
            end
            if is_block_end
                dchi::Float64 = chi_change(overlaps, candidate_overlaps)
                println("  CHI CHANGE = $(dchi)   (chi = $(candidate_maxbond) -> " *
                        "chi' = $(rtm_maxbond))")
                println(io, "# sweep $(sweep): CHI CHANGE = $(dchi) " *
                            "(chi = $(candidate_maxbond) -> chi' = $(rtm_maxbond))")
                if criteria_met && dchi < conv_threshold
                    println("Check block (block $(n_step)) CONFIRMED convergence at " *
                            "sweep $(sweep). Stopping.")
                    break
                end
                converged_flag = false
                candidate_overlaps = ComplexF64[]
                println("Check block (block $(n_step)) did NOT confirm convergence at " *
                        "sweep $(sweep) (CUT SPREAD = $(spread), SWEEP CHANGE = " *
                        "$(change), CHI CHANGE = $(dchi)); continuing with increasing " *
                        "rtm_maxbond.")
                if sweep >= max_sweeps_safety
                    println("Reached the bond-dimension ceiling ($(RTM_MAXBOND_CEILING), " *
                            "rtm_maxbond = $(rtm_maxbond)) at the end of the failed " *
                            "check block $(n_step) (sweep $(sweep)); stopping.")
                    break
                end
            end
        elseif is_block_end
            if criteria_met
                converged_flag = true
                candidate_overlaps = copy(overlaps)
                candidate_maxbond  = rtm_maxbond
                println("Converged at the end of block $(n_step) (sweep $(sweep), " *
                        "CUT SPREAD = $(spread), SWEEP CHANGE = $(change), " *
                        "threshold = $(conv_threshold)). Running one more full block " *
                        "($(rtm_maxbond_period) sweeps) as a check...")
            elseif sweep >= max_sweeps_safety
                println("Reached the bond-dimension ceiling ($(RTM_MAXBOND_CEILING), " *
                        "rtm_maxbond = $(rtm_maxbond)) at the end of block $(n_step) " *
                        "(sweep $(sweep)) without meeting both convergence criteria; " *
                        "stopping.")
                break
            end
        end

        prev_overlaps = overlaps
    end

    if !converged_flag
        println("WARNING: run stopped without confirmed convergence (bond-dimension " *
                "ceiling reached); the " *
                "singular-value spectra file reflects the LAST sweep performed " *
                "(sweep $(sweep)), not a confirmed post-convergence check block.")
    end
    write_rtm_singular_values(sv_filename, sv_spectra, steps, ref_cut, sweep,
                              converged_flag, rtm_maxbond, bitstring, n_sites_fw, steps_fw,
                              conv_threshold)
end

# CSV of the last sweep: for every cut, the full spectrum of its max-entropy bond, one row per
# singular value, with Σσ² and Rényi-2 of that cut; prints a short summary.
function write_rtm_singular_values(path::String,
                                   sv_spectra::Vector{Vector{Float64}},
                                   steps::Int,
                                   ref_cut::Int,
                                   sweep::Int,
                                   converged::Bool,
                                   rtm_maxbond::Int,
                                   bitstring::Vector{Int},
                                   n_sites_fw::Int,
                                   steps_fw::Int,
                                   conv_threshold::Float64)
    n_values::Int = sum(length, sv_spectra)
    open(path, "w") do io
        println(io, "# RTM singular-value spectra — LAST sweep only, one row per singular value")
        println(io, "# sweep_written = $(sweep)")
        println(io, "# total_sweeps_performed = $(sweep)")
        println(io, "# converged = $(converged)")
        println(io, "# conv_threshold = $(conv_threshold)")
        println(io, "# rtm_maxbond_last_sweep = $(rtm_maxbond)")
        println(io, "# L_n_sites_fw = $(n_sites_fw)")
        println(io, "# T_steps_fw = $(steps_fw)")
        println(io, "# bitstring = $(join(bitstring))")
        println(io, "# n_cuts = $(steps)")
        println(io, "# cut labelling convention: cut = k, the same 1-based spatial-cut " *
                     "index k used in the .dat file's 'k/steps' column; cut_index = k " *
                     "(the two coincide in this codebase's convention — both are given " *
                     "so Python can group by cut_index without knowing the convention)")
        println(io, "# central cut: ref_cut = cld(steps, 2) = $(ref_cut) (ties broken by " *
                     "rounding up; the cut of the reported amplitude A_{N/2}); is_central = 1 " *
                     "iff cut == ref_cut, else 0")
        println(io, "# each cut's spectrum is the FULL singular-value vector of the ONE " *
                     "internal temporal bond with the MAXIMUM spectrum entropy within " *
                     "that cut (the S returned by svd_linear at the accepted rtm_maxbond " *
                     "for that bond — no further truncation); ties keep the first bond " *
                     "encountered (strict > comparison)")
        println(io, "#")
        println(io, "cut,cut_index,is_central,n_cuts,rtm_maxbond,rank,singular_value,sumsq,renyi2")
        for k in 1:steps
            is_central::Int = k == ref_cut ? 1 : 0
            sumsq::Float64  = sumsq_singular_values(sv_spectra[k])
            renyi2::Float64 = renyi2_from_singular_values(sv_spectra[k])
            for (rank, s) in enumerate(sv_spectra[k])
                println(io, "$k,$k,$is_central,$steps,$rtm_maxbond,$rank," *
                            "$(@sprintf("%.16e", s)),$(@sprintf("%.16e", sumsq))," *
                            "$(@sprintf("%.16e", renyi2))")
            end
        end
    end
    sumsq_per_cut::Vector{Float64} = [sumsq_singular_values(sv_spectra[k]) for k in 1:steps]
    println("INFO: Σσ² central cut (k=$(ref_cut)) = $(sumsq_per_cut[ref_cut]);  " *
            "max over cuts = $(maximum(sumsq_per_cut)) at cut $(argmax(sumsq_per_cut));  " *
            "Rényi-2 central = $(renyi2_from_singular_values(sv_spectra[ref_cut]))")
    println("INFO: wrote RTM singular-value spectra to $(path) " *
             "($(steps) cuts, $(n_values) singular values total)")
end

end
