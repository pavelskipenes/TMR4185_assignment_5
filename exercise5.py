# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:59:53 2024

@author: Bruker
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

m1 = 10
m2 = 19
k1 = 12
k2 = 3
k3 = m2/2
c1 = 0.2
c2 = 0.2
c3 = 0.2
F1_abs = 15
F2_abs = 10
F1_phase = 0
F2_phase = 0
omega = 1
t = np.linspace(0, 30, 100)


def twocarts(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, F1_phase, F2_abs, F2_phase, omega):

    M = np.array([[m1, 0],
                 [0, m2]])

    C = np.array([[c1 + c2, -c2],
                 [-c2, c2 + c3]])

    K = np.array([[k1 + k2, -k2],
                 [-k2, k2 + k3]])

    F1 = F1_abs * np.exp(1j * F1_phase)  # Komplekse krefter
    F2 = F2_abs * np.exp(1j * F2_phase)
    F = np.array([F1, F2])

    # Omegamatrise
    omega2 = omega**2

    # Løsning for x1 og x2 ved bruk av komplekse ligninger
    A = -omega2 * M + 1j * omega * C + K  # Systemmatrise
    X = np.linalg.solve(A, F)  # Løs for forskyvning

    # Amplitudene er de absolutte verdiene av de komplekse forskyvningene
    x1 = np.real(X[0] * np.exp(1j * omega * t))  # Reelle delen av x1(t)
    x2 = np.real(X[1] * np.exp(1j * omega * t))

    x1_abs = np.abs(X[0])
    x2_abs = np.abs(X[1])

    return x1, x2, x1_abs, x2_abs


x1, x2, x1_abs, x2_abs = twocarts(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, F1_phase, F2_abs, F2_phase, omega)


def natural_frequency(m1, m2, k1, k2, k3):

    M = np.array([[m1, 0],
                 [0, m2]])

    K = np.array([[k1 + k2, -k2],
                 [-k2, k2 + k3]])

    D, V = la.eig(K, M)
    D = np.real(D)
    V = np.real(V)

    return D, V


D, V = natural_frequency(m1, m2, k1, k2, k3)

w_1 = np.sqrt(D[0])
w_2 = np.sqrt(D[1])
print(w_1)
print(w_2)

print(D)
print(V)

w = np.linspace(0.1, 5, 11)


def plot_frequencies(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, F1_phase, F2_abs, F2_phase):
    plt.figure(figsize=(10, 6))
    for i in w:
        x1, x2, a, b = twocarts(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, F1_phase, F2_abs, F2_phase, i)
        plt.plot(t, x1, label=i)
        # plt.plot(t, x2, label=i)
    plt.xlabel("Tid (s)")
    plt.ylabel("Forskyvning (m)")
    plt.title("Displacement with load frequencies between 0.1 to 5 [rad/sec], for m_1 (nat. freq: 1.24)")
    plt.legend()
    plt.grid(True)
    plt.savefig("m_1_plot.png")

    plt.figure(figsize=(10, 6))
    for i in w:
        x1, x2, a, b = twocarts(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, F1_phase, F2_abs, F2_phase, i)
        # plt.plot(t, x1, label=i)
        plt.plot(t, x2, label=i)
    plt.xlabel("Tid (s)")
    plt.ylabel("Forskyvning (m)")
    plt.title("Displacement with load frequencies between 0.1 to 5 [rad/sec], for m_2 (nat. freq: 0.77)")
    plt.legend()
    plt.grid(True)

    plt.savefig("m_2_plot.png")
    return 0


print(plot_frequencies(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, F1_phase, F2_abs, F2_phase))

plt.figure(figsize=(10, 6))
plt.plot(t, x1, label="x1(t)")
plt.plot(t, x2, label="x2(t)")
plt.axhline(y=x1_abs, color='blue', linestyle='--', label=x1_abs)
plt.axhline(y=x2_abs, color='orange', linestyle='--', label=x2_abs)
plt.xlabel("Tid (s)")
plt.ylabel("Forskyvning (m)")
plt.title("2b, x1 og x2")
plt.legend()
plt.grid(True)
plt.savefig("2b.png")

theta = np.linspace(0, 2*np.pi, 11)
print(theta)

plt.figure(figsize=(10, 6))
for i in theta:
    x_1, x_2, a, b = twocarts(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, F1_phase, F2_abs, i, omega)
    plt.plot(t, x_1, label=round(i, 2))
    plt.xlabel("Tid (s)")
    plt.ylabel("Forskyvning (m)")
plt.title("Response for m_1, phase [rad]")
plt.legend()
plt.grid(True)
plt.savefig("2e_m_1.png")
plt.clf()

for i in theta:
    x_1, x_2, a, b = twocarts(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, i, F2_abs, 0, omega)
    plt.plot(t, x_2, label=round(i, 2))
    plt.xlabel("Tid (s)")
    plt.ylabel("Forskyvning (m)")
plt.title("Response for m_2, phase [rad]")
plt.legend()
plt.grid(True)
plt.savefig("2e_m_2.png")

theta = np.linspace(w_1, w_2, 11)
print(theta)
