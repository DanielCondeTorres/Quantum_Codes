"""
Quantum Fourier Transform (QFT) Matrix Generator and Analysis

This script constructs the QFT matrix for a 5-dimensional quantum system (not 5 qubits),
i.e., a 5×5 matrix using complex roots of unity. It then analyzes this matrix to count 
how many entries are purely real or purely imaginary.

- The QFT matrix is built using the formula:
    QFT[i,j] = ω^(i*j) / sqrt(N)
  where ω = exp(2πi / N) is the primitive N-th root of unity.

- For N=5, this results in a 5×5 complex unitary matrix.

- After building the matrix, we count how many of its entries:
    - Have only a real component (imaginary part ≈ 0)
    - Have only an imaginary component (real part ≈ 0)

Useful for studying QFT in arbitrary dimensions, such as in qutrit or qudit systems.
"""
import numpy as np

N = 5
omega = np.exp(2j * np.pi / N)

# Construye la matriz QFT de dimensión 5x5
qft5 = np.array([[omega**(j*k) for k in range(N)] for j in range(N)]) / np.sqrt(N)

print("QFT matrix (5x5):")
print(np.round(qft5, 4))  # redondeado para visualizar mejor

# Cuenta entradas reales puras
real_entries = np.isclose(np.imag(qft5), 0)
imag_entries = np.isclose(np.real(qft5), 0)

num_real = np.sum(real_entries & ~imag_entries)
num_imag = np.sum(imag_entries & ~real_entries)

print(f"Purely real entries: {num_real}")
print(f"Purely imaginary entries: {num_imag}")
