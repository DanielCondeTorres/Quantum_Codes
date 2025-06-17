from qiskit import QuantumCircuit
from qiskit.circuit.library import QFT
import numpy as np
import math as mt
from qiskit.quantum_info import Operator

def myQFT(qubits):
    """
    Implements Quantum Fourier Transform using Qiskit.
    
    Args:
        qubits: Number of qubits for the QFT circuit
        
    Returns:
        QuantumCircuit: The QFT circuit
    """
    # Create a quantum circuit with the specified number of qubits
    circuit = QuantumCircuit(qubits)
    
    # Apply Hadamard gates and controlled phase rotations
    for i in range(qubits):
        # Apply Hadamard
        circuit.h(i)
        
        # Apply controlled phase rotations
        phase_divisor = 4  # 4,8,16,...
        for j in range(i+1, qubits):
            circuit.cp(2*np.pi/phase_divisor, j, i)
            phase_divisor = 2 * phase_divisor
    
    # Swap the qubits
    for j in range(qubits//2):  # integer division
        circuit.swap(j, qubits-j-1)
    
    return circuit

# Example usage:
if __name__ == "__main__":
    import numpy as np
    # Parameters
    n = 3
    N = 2**n
    phi = 2 * np.pi / N
    coefficient = 1/(N**0.5)
    omega = complex(mt.cos(phi), mt.sin(phi))

    # Print QFT matrix directly calculated by Python
    print("QFT matrix directly calculated by Python:")
    for i in range(N):
        row_str = ""
        for j in range(N):
            val = coefficient*omega**(i*j)
            R = round(val.real, 2)
            I = round(val.imag, 2)
            row_str += f"{R}+i({I})  "
        print(row_str)

    import numpy as np
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    # Suponemos que ya tienes definida la función myQFT(n)

    n = 3  # número de qubits (ejemplo)
    qc = myQFT(n)

    # Agrega la instrucción para guardar la matriz unitaria
    qc.save_unitary()

    # Prepara el simulador con método 'unitary'
    sim = AerSimulator(method='unitary')

    # Transpila el circuito para el simulador
    qc = transpile(qc, sim)

    # Ejecuta directamente el circuito (sin assemble())
    result = sim.run(qc).result()

    # Extrae la matriz unitaria
    unitary = result.get_unitary(qc)

    # Muestra la matriz completa
    print("Matriz unitaria de myQFT(n):\n", np.round(unitary, 5))

    # Opcional: imprimir en formato legible
    print('QFT(n) Unitary Matrix:')
    for row in unitary:
        print("  ".join(f"{round(val.real,2)}+i{round(val.imag,2)}" for val in row))
    #Directamente usando Qiskit
    qc_direct= QFT(num_qubits=n, inverse=True, do_swaps=True)
    print(Operator(qc_direct).data)
