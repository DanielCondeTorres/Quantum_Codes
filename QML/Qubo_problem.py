# En este problema tenemos 6 nodos (x1,x2,x3,x4,x5,x6) donde:
#x1 tiene 3 enlaces con x2,x6 y x5
#x2 tiene 1 enlace con x1
#x3 tiene 1 enlaces con x4
#x4 tiene 2 enlaces con x3 y x5
#x5 tiene 3 enlaces con x2, x1 y x4
#x6 tiene 1 enlacenla con x1
#La idea del problema es encontrar que nodos son los mejores para ser contratados por una empresa debido a que pueden vender muy bien un determinado producto debido a sus contactos. Como dos nodos que esten conectados entre si, de contratar  a los dos se aplica una penalización pues no van a vender a un mismo miembro de la empresa (penalty=-2)

import qubovert
coefs = [3,2,1,2,3,1,-2,-2,-2,-2,-2,-2]
qubo = qubovert.QUBO()
for i in range(6):
    qubo.create_var(f"x{i+1}")
    qubo[(f"x{i+1}",)] = coefs[i]
qubo[("x1","x6")] = -2
qubo[("x1","x2")] = -2
qubo[("x1","x5")] = -2
qubo[("x4","x5")] = -2
qubo[("x2","x5")] = -2
qubo[("x3","x4")] = -2
qubo
# Solución con un quantum annealing
anneal_res = qubovert.sim.anneal_pubo(-qubo, num_anneals=1000)
print('Solucion: ',anneal_res.best.state)
