import random
import sympy as sp
from sympy import pprint, I

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


def is_normalised(v: sp.Matrix) -> bool:
    return v.norm() == 1


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


def get_prob(state: sp.Matrix, value: int) -> sp.Expr:
    if not is_normalised(state):
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


def measure(state: sp.Matrix):
    assert is_normalised(state), "State is not normalised"

    p0 = get_prob(state, 0)
    return 0 if random.random() < eval(p0) else 1


def main():
    # state = sp.Matrix([1, 0])
    state = MINUS
    print(state)
    print(get_prob(state, 1))


if __name__ == "__main__":
    main()
