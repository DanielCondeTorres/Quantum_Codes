from cmath import exp
from math import pi, sqrt
import matplotlib.pyplot as plt

def compute_and_plot_dft_from_list(x):
    """
    Computes and visualizes the Discrete Fourier Transform (DFT) of a list.

    Parameters:
    - x: List of real or complex numbers

    Returns:
    - y: List of complex DFT values
    """
    N = len(x)
    y = []

    # Compute DFT with negative exponent
    for k in range(N):
        s = 0
        for j in range(N):
            s += exp(-2 * pi * 1j * j * k / N) * x[j]
        s *= 1 / sqrt(N)  # Optional normalization
        y.append(s)

    # Prepare data for plotting
    magnitudes = [abs(val) for val in y]
    phases = [val.real if abs(val) < 1e-6 else (val/abs(val)).imag for val in y]  # to suppress noise
    frequencies = list(range(N))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.suptitle("Discrete Fourier Transform (DFT)", fontsize=16, fontweight='bold')

    # Magnitude plot
    plt.subplot(2, 1, 1)
    plt.stem(frequencies, magnitudes, basefmt=" ", linefmt='tab:blue', markerfmt='bo')
    plt.title("Magnitude Spectrum", fontsize=12)
    plt.xlabel("Frequency Index (k)")
    plt.ylabel("Magnitude")
    plt.grid(True)

    # Phase plot
    plt.subplot(2, 1, 2)
    plt.stem(frequencies, [val.imag for val in y], basefmt=" ", linefmt='tab:red', markerfmt='ro')
    plt.title("Phase Spectrum (Imaginary Part)", fontsize=12)
    plt.xlabel("Frequency Index (k)")
    plt.ylabel("Phase")
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    return y
