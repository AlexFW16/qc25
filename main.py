import itertools
import random
import sympy as sp
from sympy import cos, exp, pprint, I, sin
from sympy import pi as PI
import numpy as np
from sympy.core.evalf import evalf
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations

sqrt2 = sp.sqrt(2)

# Basis States
ZERO = sp.Matrix([1, 0])  # Zero-State
ONE = sp.Matrix([0, 1])  # One-State
PLUS = sp.Matrix([1, 1]) / sqrt2  # Superposition Plus
MINUS = sp.Matrix([1, -1]) / sqrt2  # Superposition Minus
IPLUS = sp.Matrix([1, I]) / sqrt2  # Superposition Plus i 
IMINUS = sp.Matrix([1, -I]) / sqrt2 # Superposition Minus i 


# Basic Quantum Gates

ID = sp.eye(2)  # Identity
Z = sp.Matrix([[1, 0], [0, -1]])  # signflip
X = sp.Matrix([[0, 1], [1, 0]])  # NOT / bitflip
H = sp.Matrix([[1, 1], [1, -1]]) / sqrt2  # Hadamard
S  = sp.Matrix([[1, 0], [0, I]]) # sqrt of Z
T  = sp.Matrix([[1, 0], [0, exp( I * PI / 4)]]) # sqrt of S
Y = sp.Matrix([[0, - I], [I, 0]])

CNOT = sp.Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CNOT2 = sp.Matrix([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


# X-Axis Pauli Rotation
def R_X(theta: sp.Expr | sp.Rational | float) -> sp.Matrix:
    return sp.Matrix([[cos(theta/2), - I * sin(theta/2)], [- I * sin(theta/2), cos(theta/2)]])

# Y-Axis Pauli Rotation
def R_Y(theta: sp.Expr | sp.Rational | float) -> sp.Matrix:
    return sp.Matrix([[cos(theta/2), -sin(theta/2)], [sin(theta/2), cos(theta/2)]])

# Z-Axis Pauli Rotation
def R_Z(theta: sp.Expr | sp.Rational | float) -> sp.Matrix:
    return sp.Matrix([[exp(-I * theta / 2), 0], [0, exp(I * theta/2)]])

def commute(A: sp.Matrix, B: sp.Matrix) -> bool:
    """
    Returns true if matrices commute.
    """
    if isinstance(A, sp.Matrix) and isinstance(B, sp.Matrix):
        return A*B == B*A
        # commutator =  A*B - B*A
        # return commutator == sp.zeros(*commutator.shape())

    #NOTE: possible right now because input can only be matrix
    elif isinstance(A, np.ndarray) and isinstance(B, np.ndarray):
        commutator = np.dot(A, B) - np.dot(B, A)
        return np.allclose(commutator, np.zeros_like(commutator))
    else:
        raise TypeError("A and B must either be sympy matrices or numpy arrays, no mixing!")


def kronecker(A: sp.Matrix, B: sp.Matrix) -> sp.Matrix:
    a_shape, b_shape = A.shape, B.shape
    C : sp.Matrix = sp.zeros(a_shape[0] * b_shape[0], a_shape[1] * b_shape[1])

    for i in range (A.rows):
        for j in range(A.cols):
            # Puts the Block Matrix a_i,j * B into C, uses the starting index (0,0)-element of block matrix
            C[i * a_shape[0], j * a_shape[1]] = A[i, j] * B
    return C
            


def is_normalised(v: sp.Matrix, numerically: bool = False ) -> bool:
    norm = v.norm()
    if numerically:
        norm = float(norm.evalf())
        return bool(np.isclose(norm, 1.0))
    else:
        return sp.simplify(norm) == 1

def is_normalised_prob(v: sp.Matrix) -> float:
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

def get_probs(state: sp.Matrix, numerically: bool = False, as_cdf: bool = False) -> list[sp.Expr]:
    assert is_normalised(state, numerically), f"[Warn] {is_normalised_prob(state)}"
    
    out: list[sp.Expr] = []
    for i in range(state.rows):
        entry = state[i, 0]
        assert isinstance(entry, sp.Expr), f"Excpected expression and not {type(entry)}"

        prev = out[i-1] if i > 0 else 0
        out.append(prev + sp.Pow(abs(entry) , 2))
    return out

def get_prob_single(state: sp.Matrix, value: int, numerically: bool = False) -> sp.Expr:
    if not is_normalised(state, numerically):
        print(f"[Warn] {is_normalised_prob(state)}")
    if value not in (0, 1):
        raise ValueError("Value must be 0 or 1")
    entry = sp.sympify(state[value, 0])
    # entry = sp.sympify(state[0, 0]) if value == 0 else sp.sympify(state[0, 1])

    if not isinstance(entry, sp.Expr):
        raise TypeError(f"Excpected expression and not {type(entry)}")
    return sp.Pow(abs(entry), 2)


def compose_gates(gates: list[sp.Matrix]):
    """
    The gates are applied as if you write out the list from left [0] to right [-1]
    """
    result = sp.eye(2)
    for i in range(len(gates)):
        result = gates[i] * result
    return result

def measure(state: sp.Matrix, numerically: bool = False):
    size = state.shape[0]
    assert size % 2 == 0, "Invalid state given: Length not a power of 2"
    cdf = get_probs(state, numerically, as_cdf=True)

    r = random.random()
    for i in range(size):
        if r <= cdf[i]:
            return i

def main():
    A = sp.Matrix([[1, 0], [0, 1]])
    B = sp.Matrix([[1, 1], [1, 1]])

    G0 = kronecker(ID, ID)
    G1 = kronecker(X, ID) #bitflip first
    G2 = kronecker(ID, X) # bitflip second
    G3 = kronecker(X, X)

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
            M : sp.Matrix = sp.eye(4)
            for op in seq:
                M = M * op
            results.add(M.as_immutable())

    print(len(results))

    exit(0)
    data = []
    pprint(kronecker(H *T, H) * kronecker(ZERO, ONE))
    for i in range(100):
        # data.append(measure(kronecker(H *T, H) * kronecker(ZERO, ONE)))
        # data.append(measure((kronecker(S, ID)* CNOT2* kronecker(H, H) * CNOT * kronecker(H, H) * CNOT * kronecker(ONE, ZERO))))

        # data.append(measure(R_Z(PI / 2) * ONE))
        data.append(measure(S * ONE))

    counts = Counter(data)

    # Extract numbers and their frequencies
    numbers = list(counts.keys())
    frequencies = list(counts.values())

    # Create bar plot
    plt.bar(numbers, frequencies, width=0.0001)

    # Add labels and title
    plt.xlabel('Number')
    plt.ylabel('Occurrences')
    plt.title('Number of Occurrences')

    # Display the plot
    plt.show()























if __name__ == "__main__":
    main()
