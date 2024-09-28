from dataclasses import dataclass


@dataclass
class Input:
    m1: float
    m2: float
    k1: float
    k2: float
    k3: float
    c1: float
    c2: float
    c3: float
    F1_abs: float
    F2_abs: float
    F1_phase: float
    F2_phase: float
    omega: float


def twocarts(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, F1_phase, F2_abs, F2_phase, omega):
    pass
