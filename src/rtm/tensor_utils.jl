# tensor_utils.jl — identity and Pauli matrices used to build the gates.

module TensorUtils

using LinearAlgebra

export PAULI_MATRICES

const PAULI_MATRICES = (
    Id2 = Matrix{ComplexF64}(I, 2, 2),
    Id4 = Matrix{ComplexF64}(I, 4, 4),
    σx  = ComplexF64[0 1; 1 0],
    σz  = ComplexF64[1 0; 0 -1],
)

end
