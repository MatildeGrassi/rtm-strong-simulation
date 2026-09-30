# linear_svd.jl — SVD of an ITensor truncated to the largest maxdim singular values only
# (no cutoff; zero singular values are kept), so the RTM bond dimension is set by maxdim.

using ITensors, LinearAlgebra

# SVD of a dense 2-index tensor: LAPACK divide-and-conquer, falling back to QR iteration and
# then to the recursive SVD (the ITensors default chain). Keeps the first maxdim values.
function svd_linear_internal(T::ITensors.NDTensors.DenseTensor{ElT, 2, IndsT};
                             maxdim::Int) where {ElT, IndsT}
    ND = ITensors.NDTensors
    M  = ND.matrix(T)
    MUSV = ND.svd_catch_error(M; alg = LinearAlgebra.DivideAndConquer())
    isnothing(MUSV) && (MUSV = ND.svd_catch_error(M; alg = LinearAlgebra.QRIteration()))
    isnothing(MUSV) && (MUSV = ND.svd_recursive(M))

    MU, MS, MV = MUSV
    conj!(MV)

    dS::Int = min(maxdim, length(MS))
    if dS < length(MS)
        resize!(MS, dS)
        MU = ND.expose(MU)[:, 1:dS]
        MV = ND.expose(MV)[:, 1:dS]
    end

    u      = eltype(IndsT)(dS)
    v      = eltype(IndsT)(dS)
    Uinds  = IndsT((ND.ind(T, 1), u))
    Sinds  = IndsT((u, v))
    Vinds  = IndsT((ND.ind(T, 2), v))

    U = ND.tensor(ND.Dense(vec(MU)), Uinds)
    S = ND.tensor(ND.Diag(MS),       Sinds)
    V = ND.tensor(ND.Dense(vec(MV)), Vinds)
    return U, S, V
end

# ITensor interface: A has two indices and is split between the first and the second one.
function svd_linear(A::ITensor; maxdim::Int)
    U, S, V = svd_linear_internal(ITensors.NDTensors.tensor(A); maxdim=maxdim)
    return ITensor(U), ITensor(S), ITensor(V)
end
