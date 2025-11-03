import random
import sympy as sp
from sympy import cos, exp, pi, pprint, I, sin
import numpy as np
from sympy.core.evalf import evalf

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
T  = sp.Matrix([[1, 0], [0, exp( I * pi / 4)]]) # sqrt of S
Y = sp.Matrix([[0, - I], [I, 0]])


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


def get_prob(state: sp.Matrix, value: int, numerically: bool = False) -> sp.Expr:
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
    assert is_normalised(state, numerically), "State is not normalised"

    p0 = get_prob(state, 0, numerically)
    return 0 if random.random() < eval(p0) else 1


def main():

    print(commute(Y, Z))
    print(commute(T, S))
    print(commute(H * Z * H, R_X(-sp.pi / 2)))

    print(commute(X,    Y))


if __name__ == "__main__":
    main()
