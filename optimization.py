import cvxpy as cp
from dataset import ValorantClipDataset

n = ValorantClipDataset.__len__

import numpy as np
P = np.random.randint(2, size=(n, n))

x = cp.Variable(n)

objective = cp.Maximize(cp.sum(cp.multiply(P, x)))

constraints = [0 <= x, x <= 1]
epsilon = 0.01
for i in range(n):
    for j in range(n):
        if P[i, j] == 1:
            constraints.append(x[i] >= x[j] + epsilon)

problem = cp.Problem(objective, constraints)

result = problem.solve()

print(x.value)