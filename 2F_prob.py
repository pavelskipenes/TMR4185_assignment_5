import exercise5
import matplotlib.pyplot as plt
import numpy as np

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
F1_phase_array = np.linspace(-180, 180, 360)
F2_phase = 0
omega = 1
t = np.linspace(0, 30, 360)

f1_factor = 2*np.exp(-omega) * np.abs(np.sin(np.pi*omega))
F1_abs = F1_abs * f1_factor


def newcarts(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, F1_phase, F2_abs, F2_phase, omega):

    M = np.array([[m1, 0],
                 [0, m2]])

    C = np.array([[c1 + c2, -c2],
                 [-c2, c2 + c3]])

    K = np.array([[k1 + k2, -k2],
                 [-k2, k2 + k3]])

    # Omegamatrise
    omega2 = omega**2
    x1_list = []
    x2_list = []

    for i in range(len(F1_phase)):
        F1 = F1_abs * np.exp(1j * F1_phase[i])  # Komplekse krefter
        F2 = F2_abs * np.exp(1j * F2_phase)
        F = np.array([F1, F2])
        # Løsning for x1 og x2 ved bruk av komplekse ligninger
        A = -omega2 * M + 1j * omega * C + K  # Systemmatrise
        X = np.linalg.solve(A, F)  # Løs for forskyvning
        # Amplitudene er de absolutte verdiene av de komplekse forskyvningene
        x1 = np.real(X[0] * np.exp(1j * omega * t[i]))  # Reelle delen av x1(t)
        x1_list.append(x1)
        x2 = np.real(X[1] * np.exp(1j * omega * t[i]))
        x2_list.append(x2)

    return x1_list, x2_list


x1, x2, x1_abs, x2_abs = exercise5.twocarts(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, F1_phase, F2_abs, F2_phase, omega)
x1_var, x2_var = newcarts(m1, m2, k1, k2, k3, c1, c2, c3, F1_abs, F1_phase_array, F2_abs, F2_phase, omega)


plt.figure(figsize=(10, 6))
plt.plot(t, x1, label="x1(t)")
plt.plot(t, x2, label="x2(t)")
plt.plot(t, x1_var, label="x1_var(t)")
plt.plot(t, x2_var, label="x2_var(t)")
plt.xlabel("Tid (s)")
plt.ylabel("Forskyvning (m)")
plt.title("2F, phasebestemt F1")
plt.legend()
plt.grid(True)
plt.savefig("2F.svg")
