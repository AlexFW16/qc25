import sympy as sp
from sympy import N,  Matrix as M
from functools import reduce
from typing import cast

def kronecker(states: list[M]) -> M:
    assert len(states) > 0, "Cannot compute kronecker product of empty list of matrices"
    if len(states) == 1:
        return states[0]
    return reduce(kronecker2, states)

def kronecker2(A: M, B:M) -> M:
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    C = sp.zeros(a_rows * b_rows, a_cols * b_cols)
    for i in range(a_rows):
        for j in range(a_cols):
            row_start = i * b_rows
            col_start = j * b_cols

            for b_i in range(b_rows):
                for b_j in range(b_cols):
                    C[row_start + b_i, col_start + b_j] = cast(sp.Expr, A[i, j]) * cast(sp.Expr, B[b_i, b_j])
    return C


def is_orthonormal(vectors: list[M]) -> bool:
    shape = vectors[0].shape

    if shape[0] == 1:
        for i, vec in enumerate(vectors):
            if (vec  * vec.H)[0, 0] != [1]:
                return False
            for j in range(i+1, len(vectors)):
                if (vec * vectors[j].H)[0, 0] != 0:
                    return False
    elif shape[1] == 1:
        for i, vec in enumerate(vectors):
            if (vec.H * vec)[0, 0] != 1:
                return False
            for j in range(i+1, len(vectors)):
                if (vec.H* vectors[j])[0, 0] != 0:
                    return False
    else:
        print(f"{vectors[0]} is not in vector shape: {shape}!")
        return False
    return True

def commute(A: M, B: M) -> bool:
    """
    Returns true if matrices commute.
    """
    if isinstance(A, M) and isinstance(B, M):
        return A*B == B*A
        # commutator =  A*B - B*A
        # return commutator == sp.zeros(*commutator.shape())

    #NOTE: possible right now because input can only be matrix
    elif isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        commutator = np.dot(A, B) - np.dot(B, A)
        return np.allclose(commutator, np.zeros_like(commutator))
    else:
        raise TypeError("A and B must either be sympy matrices or numpy arrays, no mixing!")


