# Tarea 8: Ejecutar VQE (Variational Quantum Eigensolver) para estimar
# la energía del estado fundamental del Hamiltoniano:
#
# H = 2Y₀Y₁ + Z₀X₁ - X₀Z₁
#
# Paso 1: Interpretación del Hamiltoniano:
# - Este Hamiltoniano está expresado como una combinación lineal de productos de operadores de Pauli
#   actuando sobre dos qubits: qubit 0 y qubit 1.
#
# Paso 2: Se nos proporciona un "ansatz" (una forma parametrizada del estado cuántico).
# - El ansatz puede estar compuesto por compuertas como Ry, Rx y CNOT que dependen de parámetros variables.
# - Este circuito parametrizado genera un estado |ψ(θ)⟩ en función de los parámetros θ.
#
# Paso 3: Usamos VQE:
# - El algoritmo VQE usa un optimizador clásico (como COBYLA, SPSA o Nelder-Mead) para minimizar
#   el valor esperado ⟨ψ(θ)|H|ψ(θ)⟩
# - Para cada conjunto de parámetros θ, ejecutamos el circuito cuántico, medimos los términos del Hamiltoniano,
#   y calculamos el valor esperado.
# - El optimizador actualiza θ para encontrar la energía mínima.
#
# Paso 4: Resultado:
# - Al finalizar la optimización, obtenemos:
#   - θ óptimo (parámetros que generan la mejor aproximación al estado fundamental)
#   - Energía mínima estimada: ⟨ψ(θ_opt)|H|ψ(θ_opt)⟩ ≈ Energía del estado fundamental de H

print('Solution')
#Ansatz to Use, given in the exercise 
from qiskit import  QuantumCircuit
from qiskit.circuit import Parameter

def ansatz():
    qc = QuantumCircuit(2,1)
    theta_0 = Parameter('theta_0')
    theta_1 = Parameter('theta_1')
    qc.rx(theta_0,0)
    qc.cx(0,1)
    qc.ry(theta_1,1)
    return qc

# Definimos una funcion para el H, que dividiremos en H1= 2Y₀Y₁ H2=Z₀X₁ y H3=- X₀Z₁
# op will be one of Y0Y1, Z0X1, X0Z1
def measurement(op):
    qc = QuantumCircuit(2)
    #p represents the Pauli string in the circuit and i is its index
    for i, p in enumerate(op):
        if str(p) == 'X':
            qc.h(i)
        elif str(p) == 'Y':
            qc.sdg(i)
            qc.h(i)
    #We can have this since all observables act on both qubits non-trivially
    qc.cx(0,1)
    return qc


# Doing VQE
from qiskit.circuit import Parameter
from qiskit import  QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit_aer import AerSimulator
from qiskit.quantum_info import SparsePauliOp

def expectation_value(params):
    H = SparsePauliOp(["YY", "ZX", "XZ"], [2.0, 1.0, -1.0])
    e = 0
    #For each term in the Hamiltonian
    for op in H:
        qc = ansatz()
        #Asign the parameters
        bc = qc.assign_parameters(params)
        #Add the measurement circuit
        bc = bc.compose(measurement(op.paulis[0]))
        #Transpile the circuit for convenience
        bc = transpile(bc, basis_gates=['u', 'cx'])
        bc.measure(1,0)
        shots = 10000
        job = AerSimulator().run(bc,shots=shots)
        counts = job.result().get_counts(bc)
        #Compute expectation for each term in the Hamiltonian
        e += op.coeffs[0]*(-1*counts.get("1",0) / shots + 1*counts.get("0",0) / shots)
    return e


#Solution:
# Set initial parameters
params = [0.5, 0.5]
e = expectation_value(params)
print("Initial expectation value estimation:", e)
from scipy.optimize import minimize

result = minimize(expectation_value, params, method='COBYLA', options = {"maxiter" : 10000})
print("Optimized parameters:", result.x)
print("Expectation value at optimized parameters:", expectation_value(result.x))
# VQE gives us an upper bound on the ground state energy  −0.95 . This is quite far from the actual ground state energy, which is −4 . One can try different classical optimization algorithms to improve it or to increase the number of iterations of the classical optimization algorithm. But in the example above, our ansatz is not expressive enough.
