from sympy import Matrix as M, Rational
import sympy as sp
from sympy.core import I
from sympy import pi as PI
from sympy import N

sqrt2 = sp.sqrt(2)

from main import BELL00, BELLM, CNOT, H, ID, X, Z, ZERO, get_probs, kronecker, measure
class Superdense:

    # Generates a 00 bell state
    def gen_bell(self) -> M:
        return BELL00

    # Alice encodes what she wants to send
    def encode(self, b0: int, b1: int, bell_state: M) -> M:
        assert b0 in [0, 1] and b1 in [0, 1]
        return (kronecker(Z ** b0 * X ** b1, ID)) * bell_state

    # Bob receives 2 qbuits and restores the information
    def receive(self, qubits: M) -> int:
        print((get_probs(N(BELLM * qubits))))
        return measure(BELLM * qubits)

    def transmit(self, data: list[list[int]]) -> list[int]:
        out: list[int] = []
        for bits in data:
            out.append(self.receive(self.encode(bits[0], bits[1], self.gen_bell())))
        return out




def main():
    S = Superdense()
    # data = [[0, 0], [1, 1], [0, 1], [1, 0], [0, 0]]
    # print(S.transmit(data))

    print(S.receive(M([1/sqrt2, 0, 0, -1/sqrt2])))
    print(S.receive(M([sp.exp(- I * PI /3 ) /sqrt2 , 0, 0, sp.exp(-I * PI /3)/sqrt2])))
    
    psi = M([Rational(1,2) + I*Rational(1,2), 0, 0, Rational(1,2) - I*Rational(1,2)])
    print(S.receive(psi))
    print(S.receive(M([1/2, -I/2, -I/2, 1/2])))
    print(S.receive(M([1/2 + I * 1/2,0, 0, 1/2 + I * 1/2 ])))
    # print(S.receive(M([1/sqrt2,  0, 0, exp(I * PI)  * 1/sqrt2])))

if __name__ == "__main__":
    main()
