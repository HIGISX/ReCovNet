from gurobipy import *
import numpy as np
import random
import time


def gurobi_solver_MCLP(users, facilities, demand, PN, A):
    # Problem datas
    N = len(users)
    M = len(facilities)

    model = Model('MCLP')
    model.setParam('OutputFlag', False)
    model.setParam('MIPFocus', 2)
    # Add variables
    client_var = {}
    serv_var = {}

    # Add Client Decision Variables and Service Decision Variables
    for j in range(N):
        client_var[j] = model.addVar(vtype="B", name="y(%s)"%j)
    for i in range(M):
        serv_var[i] = model.addVar(vtype="B", name="x(%s)"%i)
    # Update Model Variables
    model.update()
    #     Set Objective Function
    model.setObjective(quicksum(demand[j]*client_var[j] for j in range(N)), GRB.MAXIMIZE)
    #     Add Constraints
    # Add Coverage Constraints
    for j in range(N):
        model.addConstr(quicksum (A[i,j]*serv_var[i] for i in range(M)) - client_var[j] >= 0,
                        'Coverage_Constraint_%d' % j)

    # Add Facility Constraint
    model.addConstr(quicksum(serv_var[i] for i in range(M)) == PN,
                "Facility_Constraint")
    start = time.time()
    model.optimize()
    print(f"solving time: {time.time()-start}")
    # return a stardard result list
    x_result = []
    for i in range(M):
        x_result.append(serv_var[i].X)
    y_result = []
    for j in range(N):
        y_result.append(client_var[j].X)
    obj = model.ObjVal
    return x_result, y_result, obj


n_users = 2000
n_facilities = 1000
n_centers = 15
radius = 0.15
users = [(random.random(), random.random()) for i in range(n_users)]
facilities = [(random.random(), random.random()) for i in range(n_facilities)]
demand = np.random.randint(1, 2, size=n_users)
users, facilities = np.array(users), np.array(facilities)
A = np.sum((facilities[:, np.newaxis, :] - users[np.newaxis, :, :]) ** 2, axis=-1) ** 0.5

mask1 = A <= radius
A[mask1] = 1
A[~mask1] = 0
start = time.time()
x_result, y_result, obj = gurobi_solver_MCLP(users, facilities, demand, n_centers, A)
end = time.time() - start
print(f"The objective of {n_users}-{n_facilities}-{n_centers} MCLP samples is: {obj}")
print(f"The solving time of {n_users}-{n_facilities}-{n_centers} MCLP samples is: {end}")