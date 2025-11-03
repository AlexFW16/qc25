from typing import cast
import sympy as sp
from sympy.core.evalf import evalf

from main import PLUS, R_Y, ZERO, measure

class ParityGame:

    def __init__(self, length: int, rationals: list[sp.Rational], numerically: bool = True):
        self.length = length
        self.rationals = rationals
        self.numerically = numerically


    def pass_on(self,state: sp.Matrix, input_rational: sp.Rational) -> sp.Matrix:
        """Passes the q-bit on to the next person."""
        theta: float | sp.Expr

        if self.numerically:
            theta = float(sp.pi * input_rational)
        else:
            theta = sp.pi * input_rational

        state = R_Y(theta) * state
        return state

    def play(self, state: sp.Matrix) -> int:
        if not sp.Rational(sum(self.rationals)).is_integer:
            raise KeyError("Provided list of rationals does not sum to integer!")

        for r in self.rationals:
            state = self.pass_on(state, r)

        return measure(state, self.numerically)

rationals: list[sp.Rational] = cast(list[sp.Rational], [
    sp.Rational(p, q) for p, q in [
    (1, 6),
    # (1, 2),
    # (1, 2),
    (1, 4),
    (1, 3),
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (1, 5),
    (2, 5),
    (3, 5),
    (4, 5),
    (1, 10),
    (3, 10),
    (7, 10),
    (12, 30),]

])
print(sum(rationals))  # 2
game = ParityGame(10,rationals)

outcomes = []
for i in range(100):
    outcomes.append(game.play(ZERO))

print(sum(outcomes))


