import pennylane as qml
import matplotlib.pyplot as plt

# Define the problem Hamiltonian for QAOA
# H = 0.1*Z0 - 2*Z1*Z0
H = 0.1 * qml.PauliZ(0) - 2 * qml.PauliZ(1) @ qml.PauliZ(0)

# QAOA parameters (angles for cost and mixer unitaries)
gammas = [0.3, 0.2]
betas = [0.2, 0.4]

# Set up a 2-qubit quantum device
dev = qml.device("default.qubit", wires=[0, 1])

def mixer_layer(beta, wires):
    """Applies the standard QAOA mixer layer: RX(2*beta) on each qubit."""
    for i in wires:
        qml.RX(2 * beta, wires=i)

@qml.qnode(dev)
def qaoa(H, gammas, betas):
    """
    QAOA circuit for a given Hamiltonian H and parameter lists gammas, betas.
    Args:
        H (qml.Hamiltonian): The cost Hamiltonian.
        gammas (list[float]): Angles for the cost unitary.
        betas (list[float]): Angles for the mixer unitary.
    Returns:
        array: Probabilities of measuring each computational basis state.
    """
    # Start in uniform superposition
    for i in range(2):
        qml.Hadamard(wires=i)
    # Apply alternating cost and mixer layers
    for gamma, beta in zip(gammas, betas):
        # Cost layer: time evolution under H for time gamma
        qml.templates.ApproxTimeEvolution(H, gamma, 1)
        # Mixer layer: RX rotations
        mixer_layer(beta, wires=[0, 1])
    return qml.probs(wires=[0, 1])

# Run the QAOA circuit and get output probabilities
output = qaoa(H, gammas, betas)

# Plot the output distribution
plt.bar(range(len(output)), output)
plt.xlabel('Basis state')
plt.ylabel('Probability')
plt.title('QAOA Output Distribution')
plt.show()
