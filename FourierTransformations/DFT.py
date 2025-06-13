from cmath import exp
from math import pi, sqrt
import matplotlib.pyplot as plt

def compute_and_plot_dft_from_list(x):
    """
    Computes and visualizes the Discrete Fourier Transform (DFT) of a list.

    Parameters:
    - x: List of real or complex numbers representing the input signal.

    Returns:
    - y: List of complex DFT values.
    """
    N = len(x)  # Length of input signal
    y = []      # List to store DFT results

    # Compute the DFT using the formula with negative exponent
    for k in range(N):
        s = 0
        for j in range(N):
            s += exp(-2 * pi * 1j * j * k / N) * x[j]
        s *= 1 / sqrt(N)  # Normalize output (optional)
        y.append(s)

    # Extract magnitude and phase data for plotting
    magnitudes = [abs(val) for val in y]
    phases = [val.real if abs(val) < 1e-6 else (val / abs(val)).imag for val in y]
    frequencies = list(range(N))

    real_parts = [val.real for val in y]
    imag_parts = [val.imag for val in y]

    # Plot Magnitude, Phase and Complex plane in one figure
    plt.figure(figsize=(14, 12))
    plt.suptitle("Discrete Fourier Transform (DFT) Analysis", fontsize=18, fontweight='bold')

    plt.subplot(3, 1, 1)
    plt.stem(frequencies, magnitudes, basefmt=" ", linefmt='tab:blue', markerfmt='bo')
    plt.title("Magnitude Spectrum", fontsize=14)
    plt.xlabel("Frequency Index (k)", fontsize=12)
    plt.ylabel("Magnitude", fontsize=12)
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.stem(frequencies, imag_parts, basefmt=" ", linefmt='tab:red', markerfmt='ro')
    plt.title("Phase Spectrum (Imaginary Part)", fontsize=14)
    plt.xlabel("Frequency Index (k)", fontsize=12)
    plt.ylabel("Phase", fontsize=12)
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.scatter(real_parts, imag_parts, color='purple', edgecolors='k', s=60, alpha=0.75)
    plt.axhline(0, color='grey', lw=1)
    plt.axvline(0, color='grey', lw=1)
    plt.title("DFT Values on the Complex Plane", fontsize=14)
    plt.xlabel("Real Part", fontsize=12)
    plt.ylabel("Imaginary Part", fontsize=12)
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    # Now plot the raw complex DFT values using plt.plot (real and imaginary parts)
    plt.figure(figsize=(12, 5))
    plt.plot(frequencies, real_parts, 'b-', label='Real Part')
    plt.plot(frequencies, imag_parts, 'r--', label='Imaginary Part')
    #plt.plot(y)
    plt.title("Raw DFT Values (Real and Imaginary Parts)", fontsize=16)
    plt.xlabel("Frequency Index (k)", fontsize=14)
    plt.ylabel("DFT Value", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return y
