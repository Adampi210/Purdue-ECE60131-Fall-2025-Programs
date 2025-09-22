import numpy as np

# Define probability distributions
# p(X1), p(X2|X1), p(X3|X2), p(X4|X3)
p_X1 = np.array([0.5, 0.5]).reshape((2, 1, 1, 1))
p_X2_given_X1 = np.array([[0.6, 0.2], [0.4, 0.8]]).reshape((2, 2, 1, 1))
p_X3_given_X2 = np.array([[0.6, 0.2], [0.4, 0.8]]).reshape((1, 2, 2, 1))
p_X4_given_X3 = np.array([[0.6, 0.2], [0.4, 0.8]]).reshape((1, 1, 2, 2))    

psi_3 = (p_X4_given_X3 * p_X3_given_X2).sum(axis=2, keepdims=True)
psi_2 = (psi_3 * p_X2_given_X1).sum(axis=1, keepdims=True)
psi_1 = (psi_2 * p_X1)

psi_1_and_X4_is_1 = psi_1[:, :, :, 1]
psi_1_given_X4_is_1 = psi_1_and_X4_is_1 / psi_1_and_X4_is_1.sum()

print(f"p(X1|X4=1) = {psi_1_given_X4_is_1.flatten()}")
print(f"p(X1=0|X4=1) = {psi_1_given_X4_is_1[0,0,0]}")
