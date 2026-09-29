# Approximate Strong Simulation of 1D Quantum Dynamics via Reduced Transition Matrix

Julia code accompanying the Master's thesis

> **Approximate Strong Simulation of One-Dimensional Quantum Dynamics via Reduced Transition Matrix**
> Matilde Grassi — Università di Bologna, Department of Physics and Astronomy "A. Righi", 2026
> Supervisor: Prof. Lorenzo Piroli · Co-supervisor: Prof. Jacopo De Nardis

The code computes single transition amplitudes

$$\mathcal{A}_{xy}(T) = \langle x|\,\hat U(T)\,|y\rangle$$

between two product states of a one-dimensional qubit chain, to a prescribed relative precision. The space-time tensor network is contracted **transversally** (along the spatial direction). The left and right temporal MPS are truncated on their **reduced transition matrix (RTM)**, and the cut is swept across the system until the amplitude converges.

Two TEBD codes are included as references:

- **TEBD-FW**: standard forward-in-time TEBD.
- **TEBD-TR**: transverse contraction truncated on the reduced density matrix.

Supported dynamics:

- kicked (Floquet) Ising circuits,
- random brickwork circuits (exchange-symmetric Haar-random two-qubit gates),
- Trotterised mixed-field Ising chain $H = J\sum XX + h_x\sum X + h_z\sum Z$.

Built on [ITensors.jl](https://github.com/ITensor/ITensors.jl) / [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl).

---

## Repository structure

```
rtm-strong-simulation/
├── src/
│   ├── rtm/                    # RTM transverse contraction (modular)
│   │   ├── main.jl             # entry point: parses arguments, runs the RTM sweeps
│   │   ├── rtm.jl              # RTM contraction step + adaptive sweeping / convergence
│   │   ├── linear_svd.jl       # SVD with custom (linear) truncation used by the RTM
│   │   ├── gates.jl            # gate matrices and rotated (transverse) gate layers
│   │   ├── initial_states.jl   # boundary temporal MPS (Bell pairs + fixed ends)
│   │   ├── entanglement.jl     # entropies of MPS bonds and RTM spectra
│   │   └── tensor_utils.jl     # Pauli / identity matrices
│   └── tebd/
│       ├── tebd_fw.jl          # reference: forward TEBD
│       └── tebd_tr.jl          # reference: transverse TEBD with RDM truncation
├── scripts/                    # example launch scripts (to be added)
├── Project.toml                # Julia dependencies
└── README.md
```

---

## Installation

Requires Julia ≥ 1.12.4

```bash
git clone https://github.com/<username>/rtm-strong-simulation.git
cd rtm-strong-simulation
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Dependencies: `ITensors`, `ITensorMPS`, `JLD2`, plus the standard libraries `LinearAlgebra`, `Random` and `Printf`.

---

## Common conventions

All three programs are run from the command line and share these conventions.

| Concept | Meaning |
|---|---|
| `dt` | time step |
| `hx`, `hz` | transverse / longitudinal fields (used only in Trotter mode) |
| `states` | `0`: $\|0\cdots0\rangle \to \langle 0\cdots0\|$ · `1`: $\|0\cdots0\rangle \to \langle x\|$ · `2`: $\|y\rangle \to \langle x\|$, with $x, y$ random bit strings |
| `iter` | integer seed: fixes $x$, $y$ and the random gates |
| `gate_random` | `true`: random brickwork circuit; `false`: kicked Ising gate |
| `translational_invariance` | `true`: the same random gate is used everywhere |

Output files are written to a folder relative to the directory Julia is launched from.

---

## Usage

### RTM (main method)

```bash
julia --project=. src/rtm/main.jl n_sites_fw dt steps_fw startbondRDM maxbondSweep hx hz \
      states iter gate_random translational_invariance dual_unitary [conv_threshold] [nstart]
```

| # | Argument | Description |
|---|---|---|
| 1 | `n_sites_fw` | number of physical sites $L$ (even) |
| 2 | `dt` | time step |
| 3 | `steps_fw` | number of time steps $T$ |
| 4 | `startbondRDM` | max bond dimension used to build the initial right boundary MPS |
| 5 | `maxbondSweep` | max bond dimension for gate-layer applications during the sweeps |
| 6–7 | `hx`, `hz` | fields |
| 8 | `states` | 0, 1 or 2 |
| 9 | `iter` | seed |
| 10 | `gate_random` | `true` / `false` |
| 11 | `translational_invariance` | `true` / `false` |
| 12 | `dual_unitary` | `true` / `false` |
| 13 | `conv_threshold` | *(optional, default `0.1`)* relative precision target on the amplitude |
| 14 | `nstart` | *(optional, default `1`)* first block of the RTM bond schedule $\chi = 16 + n^2$ |

**Output** (in `IsingCircuitRTM/`):

- `Prob.RTM.sweeps_*.dat`: one row per (sweep, cut), with columns
  `sweep  k/steps  Re(A)  Im(A)  |A|  cut_spread  sweep_change  maxlinkdim(L)  maxlinkdim(R)  S_RTM  Renyi2_mid`
  (lines starting with `#` are metadata).
- `rtm_singular_values_*.dat`: CSV with the RTM singular-value spectra of the last sweep, one row per singular value.

### TEBD-FW (reference)

```bash
julia --project=. src/tebd/tebd_fw.jl n_sites dt steps maxbond states iter hx hz \
      gate_random translational_invariance dual_unitary
```

Evolves $|y\rangle$ forward with bond dimension ≤ `maxbond` and, at every time step, writes the overlap with $\langle x|$, its modulus, the maximum entanglement entropy and the bond dimension.

Output: `IsingCircuit/Prob.TEBD.FW_*.dat`.

### TEBD-TR (reference)

```bash
julia --project=. src/tebd/tebd_tr.jl n_sites_fw dt steps_fw maxbond states iter hx hz \
      gate_random translational_invariance
```

Contracts the rotated network column by column, truncating on the reduced density matrix with bond dimension ≤ `maxbond`. It writes the temporal entanglement along the contraction and, at the end, the amplitude and its modulus.

Output: `IsingCircuit/Prob.TEBD.TR_*.dat`.

---

## Examples

```bash
# Kicked Ising circuit, L = 20, T = 10, random bit strings x and y, target precision 1e-2
julia --project=. src/rtm/main.jl 20 1.0 10 64 64 0.0 0.0 2 1 false true false 0.01

# Same instance with forward TEBD as a reference
julia --project=. src/tebd/tebd_fw.jl 20 1.0 10 1024 2 1 0.0 0.0 false true false
```

<!-- TODO: add the scripts in scripts/ and the parameters used for the figures of the thesis -->

---

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{Grassi2026RTM,
  author = {Grassi, Matilde},
  title  = {Approximate Strong Simulation of One-Dimensional Quantum Dynamics via Reduced Transition Matrix},
  school = {Universit\`a di Bologna},
  year   = {2026}
}
```

## License

<!-- TODO: choose a license (e.g. MIT) after checking with the supervisors -->
