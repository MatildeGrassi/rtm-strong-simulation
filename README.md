# Approximate Strong Simulation of 1D Quantum Dynamics via Reduced Transition Matrix

Julia code for the Master's thesis research

> **Approximate Strong Simulation of One-Dimensional Quantum Dynamics via Reduced Transition Matrix**
> Matilde Grassi — MSc student, Università di Bologna, Department of Physics and Astronomy "A. Righi", 2026
>
> Research carried out in collaboration with CY Cergy Paris Université and Collège de France, in the group of Prof. Jacopo De Nardis, and with Stefano Carignano and Luca Tagliacozzo.

Computes single amplitudes

$$\mathcal{A}_{xy}(T) = \langle x| \hat U(T) |y\rangle$$

by transverse contraction with truncation on the **reduced transition matrix (RTM)**.

A more detailed analysis of the circuit case can be found in:

> M. Grassi, S. Carignano, L. Tagliacozzo, J. De Nardis, *Strong Simulation of 1D Quantum Circuits via Reduced Transition Matrices*, [arXiv:2610.02082](https://arxiv.org/abs/2610.02082) (2026).

Models:

- kicked Ising circuit;
- random brick-wall circuits (symmetrised or Haar);
- Trotterised Ising chain $H = J\sum XX + h_x\sum X + h_z\sum Z$.

Built on [ITensors.jl](https://github.com/ITensor/ITensors.jl) and [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl).

---

## Three codes

Three independent codes compute the same amplitude in different ways. With the same inputs they build the same circuit and agree with each other.

- **RTM** (`src/rtm/`): transverse contraction with RTM truncation, swept until convergence. This is the main method.
- **TEBD-TR** (`src/tebd/tebd_tr.jl`): the same transverse contraction in a single pass, with standard SVD truncation.
- **TEBD-FW** (`src/tebd/tebd_fw.jl`): forward evolution in time. It is the exact reference for small $L$.

---

## Structure

```
rtm-strong-simulation/
├── src/
│   ├── rtm/
│   │   ├── main.jl             # entry point
│   │   ├── rtm.jl              # RTM truncation and sweeps
│   │   ├── linear_svd.jl       # SVD with maxdim truncation
│   │   ├── gates.jl            # rotated gate layers
│   │   ├── random_gates.jl     # random two-qubit gates
│   │   ├── initial_states.jl   # edge temporal MPS
│   │   ├── entanglement.jl     # RTM spectrum entropies
│   │   └── tensor_utils.jl     # Pauli matrices
│   └── tebd/
│       ├── tebd_fw.jl          # forward TEBD
│       └── tebd_tr.jl          # transverse TEBD
└── README.md
```

---

## Installation

Requires Julia ≥ 1.12.4.

```bash
git clone https://github.com/MatildeGrassi/rtm-strong-simulation.git
cd rtm-strong-simulation
julia -e 'using Pkg; Pkg.add(["ITensors", "ITensorMPS"])'
```

---

## Arguments

All three codes share the same arguments:

```
L T gate states iter maxbond [...] [dt hx hz]
```

| Argument | Meaning |
|---|---|
| `L` | chain length (even) |
| `T` | periods / Trotter steps |
| `gate` | gate type (see below) |
| `states` | `0`: $0 \to 0$ · `1`: $0 \to x$ · `2`: $y \to x$ |
| `iter` | seed of $x$, $y$ and random gates |
| `maxbond` | bond cap of the gate layers |
| `dt hx hz` | only for `trotter` |

| `gate` | Meaning |
|---|---|
| `kicked` | kicked Ising ($g = 0.81$, $h = 0.904508$) |
| `sym` | symmetrised random, redrawn everywhere |
| `haar` | Haar random, redrawn everywhere |
| `sym_ti`, `haar_ti` | one random gate everywhere |
| `trotter` | Trotterised Ising chain |

The same `iter` gives the same $x$, $y$ and gates in all three codes.

---

## Usage

### RTM

```bash
julia src/rtm/main.jl L T gate states iter maxbond conv_threshold nstart [dt hx hz]
```

- `conv_threshold`: tolerance $\varepsilon$.
- `nstart`: first $n$ of $\chi = 16 + n^2$.

Output in `IsingCircuitRTM/`:

- `Prob.RTM_*.dat`: one row per sweep and cut, with columns
  `sweep  k/steps  Re(A)  Im(A)  |A|  Δcut  Δsweep  χ_L  χ_R  S_SVD  Rényi2_mid`
- `rtm_singular_values_*.dat`: RTM spectra of the last sweep (CSV).

### TEBD-FW

```bash
julia src/tebd/tebd_fw.jl L T gate states iter maxbond [dt hx hz]
```

Output: `IsingCircuitTEBD/Prob.TEBD.FW_*.dat`, one row per period, with columns
`step  overlap  |overlap|  S_max  χ_max`.

### TEBD-TR

```bash
julia src/tebd/tebd_tr.jl L T gate states iter maxbond [dt hx hz]
```

Output: `IsingCircuitTEBD/Prob.TEBD.TR_*.dat`, one row at time $T$, same columns.

---

## Examples

```bash
# RTM, kicked Ising, L = 20, T = 8, random x and y, ε = 0.1
julia src/rtm/main.jl 20 8 kicked 2 1 1024 0.1 1

# RTM, Trotterised chain, δt = 0.1, chaotic regime
julia src/rtm/main.jl 20 8 trotter 2 1 1024 0.1 1 0.1 0.5 -1.05

# Exact reference (forward TEBD, χ = 2^(L/2))
julia src/tebd/tebd_fw.jl 20 8 kicked 2 1 1024
```

Regimes of the thesis (`dt hx hz`):

- gapped: `0.1 0.0 -1.5`;
- critical: `0.1 0.0 1.0`;
- chaotic: `0.1 0.5 -1.05`.

---

## References

1. M. C. Bañuls, M. B. Hastings, F. Verstraete, J. I. Cirac, *Matrix product states for dynamical simulation of infinite chains*, Phys. Rev. Lett. **102**, 240603 (2009).
3. G. Vidal, *Efficient classical simulation of slightly entangled quantum computations*, Phys. Rev. Lett. **91**, 147902 (2003).
4. S. Carignano, C. Ramos-Marimón, L. Tagliacozzo, *Temporal entropy and the complexity of computing the expectation value of local operators after a quench*, Phys. Rev. Research **6**, 033021 (2024).
5. S. Carignano, G. Lami, J. De Nardis, L. Tagliacozzo, *Overcoming the entanglement barrier with sampled tensor networks* (2025).
6. S. Carignano, *The ITransverse.jl library for transverse tensor network contractions*, SciPost Phys. Codebases 63 (2026).
7. M. Fishman, S. R. White, E. M. Stoudenmire, *The ITensor software library for tensor network calculations*, SciPost Phys. Codebases 4 (2022).

---

## Citation

If you refer to this code, please cite:

```bibtex
@mastersthesis{Grassi2026RTM,
  author = {Grassi, Matilde},
  title  = {Approximate Strong Simulation of One-Dimensional Quantum Dynamics via Reduced Transition Matrix},
  school = {Universit\`a di Bologna},
  year   = {2026}
}
```

---

## License

No license: all rights reserved. The code is shared for reference only and is not a maintained library. For any reuse, please contact the author.