import itertools
import random
import sympy as sp
from sympy import N, Matrix as M
from sympy import cos, exp, pprint, I, sin
from sympy import pi as PI
import numpy as np
from sympy.core.evalf import evalf
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

sqrt2 = sp.sqrt(2)

# math functions
def kronecker(A: M, B: M) -> M:
    a_shape, b_shape = A.shape, B.shape
    C : M = sp.zeros(a_shape[0] * b_shape[0], a_shape[1] * b_shape[1])

    for i in range (A.rows):
        for j in range(A.cols):
            # Puts the Block Matrix a_i,j * B into C, uses the starting index (0,0)-element of block matrix
            C[i * a_shape[0], j * a_shape[1]] = A[i, j] * B
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

# Basis 1-qbuit states
ZERO:M  = M([1, 0])  # Zero-State
ONE: M = M([0, 1])  # One-State
PLUS: M = M([1, 1]) / sqrt2  # Superposition Plus
MINUS: M = M([1, -1]) / sqrt2  # Superposition Minus
IPLUS: M = M([1, I]) / sqrt2  # Superposition Plus i 
IMINUS: M = M([1, -I]) / sqrt2 # Superposition Minus i 

# Basis 2-qbuit states
BELL00: M = M(1/sqrt2 * (kronecker(ZERO, ZERO) + kronecker(ONE, ONE))) # 00 Bell State
BELL01: M = M(1/sqrt2 * (kronecker(ZERO, ONE) + kronecker(ONE, ZERO))) # 01 Bell State
BELL10: M= M(1/sqrt2 * (kronecker(ZERO, ZERO) - kronecker(ONE, ONE))) # 10 Bell State
BELL11: M = M(1/sqrt2 * (kronecker(ZERO, ONE) - kronecker(ONE, ZERO))) # 11 Bell State


# Basic 1-qubit Gates
ID = sp.eye(2)  # Identity
Z = M([[1, 0], [0, -1]])  # signflip
X = M([[0, 1], [1, 0]])  # NOT / bitflip
H = M([[1, 1], [1, -1]]) / sqrt2  # Hadamard
S  = M([[1, 0], [0, I]]) # sqrt of Z
T  = M([[1, 0], [0, exp( I * PI / 4)]]) # sqrt of S
Y = M([[0, - I], [I, 0]])

# Basic 2-qubit Gates 
CNOT = M([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]) # first bit control
CNOT2 = M([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])
BELLM = kronecker(H, ID) * CNOT


# X-Axis Pauli Rotation
def R_X(theta: sp.Expr | sp.Rational | float) -> M:
    return M([[cos(theta/2), - I * sin(theta/2)], [- I * sin(theta/2), cos(theta/2)]])

# Y-Axis Pauli Rotation
def R_Y(theta: sp.Expr | sp.Rational | float) -> M:
    return M([[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]])

# Z-Axis Pauli Rotation
def R_Z(theta: sp.Expr | sp.Rational | float) -> M:
    return M([[exp(-I * theta / 2), 0], [0, exp(I * theta/2)]])

#NOTE: just use Hermite conjugate directly
# def bra_to_ket(A: M) -> M:
#     return A.H
    
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



def is_normalised(v: M, numerically: bool = False ) -> bool:
    norm = v.norm()
    if numerically:
        norm = float(norm.evalf())
        return bool(np.isclose(norm, 1.0))
    else:
        return sp.simplify(norm) == 1

def is_normalised_prob(v: M) -> float:
    """
    Returns a probablity of being normalised, to manually disregard small
    imperfections when using concrete numbers.
    """
    return eval(1 - v.norm())


def eval(expr: sp.Expr) -> float:
    """
    Evaluates a sympy expression to a float.
    """
    return float(sp.Float(expr.evalf()))

def get_probs(state: M, numerically: bool = False, as_cdf: bool = False) -> list[sp.Expr]:
    assert is_normalised(state, numerically), f"[Warn] {is_normalised_prob(state)}"
    
    out: list[sp.Expr] = []
    for i in range(state.rows):
        entry = state[i, 0]
        assert isinstance(entry, sp.Expr), f"Excpected expression and not {type(entry)}"

        prev = out[i-1] if i > 0 else 0
        out.append(prev + sp.Pow(abs(entry) , 2))
    return out

def get_prob_single(state: M, value: int, numerically: bool = False) -> sp.Expr:
    if not is_normalised(state, numerically):
        print(f"[Warn] {is_normalised_prob(state)}")
    if value not in (0, 1):
        raise ValueError("Value must be 0 or 1")
    entry = sp.sympify(state[value, 0])
    # entry = sp.sympify(state[0, 0]) if value == 0 else sp.sympify(state[0, 1])

    if not isinstance(entry, sp.Expr):
        raise TypeError(f"Excpected expression and not {type(entry)}")
    return sp.Pow(abs(entry), 2)


def compose_gates(gates: list[M]):
    """
    The gates are applied as if you write out the list from left [0] to right [-1]
    """
    result = sp.eye(2)
    for i in range(len(gates)):
        result = gates[i] * result
    return result

def measure(state: M, numerically: bool = True) -> int:
    size = state.shape[0]
    assert size % 2 == 0, "Invalid state given: Length not a power of 2"

    cdf = get_probs(state, numerically, as_cdf=True)
    if numerically:
        cdf = [float(N(x)) for x in cdf]
        assert abs(cdf[-1] -1) < 1e-12, f"Sum over CDF is not 1: {cdf[-1]}"
    else:
        assert cdf[-1] == 1, f"Sum over CDF is not 1: {cdf[-1]}"

    r = random.random()
    for i in range(size):
        if r <= cdf[i]:
            return i
    return -1

def main():
    s1 = kronecker(X * Z, ID) * BELL00
    s2 = kronecker(Z * X, ID) * BELL00
    print(get_probs(s1))
    print(get_probs(s2))
    
    print(is_orthonormal([BELL00, BELL01, BELL10, BELL11]))


    exit(0)
    state = kronecker(MINUS, ZERO)
    print(get_probs(BELLM * state))
    data = []
    for i in range(10000):
        data.append(measure(BELLM * state))
    plot(data)

    vectors = [M([1, 0, 0, 0]).transpose(), M([0, 1, 0, 0]).transpose(), M([0, 0 , 0, 1]).transpose(),  M([0, 0, 1, 0]).transpose()]
    print(vectors)
    # print(is_orthonormal([BELL00, BELL01, BELL10, BELL11]))
    print("---")
    pprint(IPLUS.H * PLUS)
    exit(0)

    A = M([[1, 0], [0, 1]])
    B = M([[1, 1], [1, 1]])

    G0: M = kronecker(ID, ID)
    G1: M = kronecker(X, ID) #bitflip first
    G2: M = kronecker(ID, X) # bitflip second
    G3: M = kronecker(X, X)

    # pprint(CNOT)
    # state = kronecker(ONE, ONE)
    # pprint(state)
    # pprint(CNOT * state)
    #

    # pprint(kronecker(T, ID) * CNOT2 * kronecker(H, H) * CNOT * kronecker(H, H) * kronecker(ONE, ZERO))
    # print(measure(CNOT * kronecker(ZERO, ZERO)))
    # print(measure(CNOT* kronecker(ZERO, ONE)))
    # print(measure(CNOT* kronecker(ONE, ZERO)))
    # print(measure(CNOT *kronecker(ONE, ONE)))
    #
    # print("---")
    # print(measure(CNOT2 * kronecker(ZERO, ZERO)))
    # print(measure(CNOT2* kronecker(ZERO, ONE)))
    # print(measure(CNOT2* kronecker(ONE, ZERO)))
    # print(measure(CNOT2 * kronecker(ONE, ONE)))
    #

    results = {sp.eye(4).as_immutable()}  # start with the identity
    ops = [CNOT, CNOT2]

    # Try all sequences of up to length 5
    for i in range(1, 6):
        for seq in itertools.product(ops, repeat=i):
            Mat : M = sp.eye(4)
            for op in seq:
                Mat = Mat * op
            results.add(M.as_immutable())

    print(len(results))

    exit(0)
def plot(data):
    counts = Counter(data)

    # Extract numbers and their frequencies
    numbers = list(counts.keys())
    frequencies = list(counts.values())

    # Create bar plot
    plt.bar(numbers, frequencies )

    # Add labels and title
    plt.xlabel('Number')
    plt.ylabel('Occurrences')
    plt.title('Number of Occurrences')

    # Display the plot
    plt.show()























if __name__ == "__main__":
    main()
