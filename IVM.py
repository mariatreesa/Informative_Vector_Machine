import Gpy
import numpy as np
import pandas as pd
from sklearn import datasets

def ivm_regression(size, data, kernel, beta):
	"""size: The desired size of the active set
	   data: The training set
	   kernel: The optimized kernel
	   beta: Precision (1/gaussian noise variance)"""
       Sigma = kern.K(data,data)
	   diag_k = np.diag(Sigma)
	   active_set = []
	   M = np.zeros((size, data.shape[0]))
	   data = pd.DataFrame(data)
	   R = data
	   for i in range(size):
	   	max_entropy = float('-inf')
	   	max_nu = 0
	   	max_j = 0
	   	for j in R:
	   		nu = 1/(diag_k[j] + beta)
	   		diff_entropy = -0.5 * np.log(1 - (nu*diag_k[j]))
	   		if diff_entropy > max_entropy:
	   			max_entropy = diff_entropy
	   			max_nu = nu
	   			max_j = j
	   	ni = max_j
	   	K_n = Sigma[ni, :]
	   	S = np.asmatrix(K_n - (np.dot(M.T, M[:, ni], )))
	   	diag_k = diag_k - (max_nu * np.diag(np.dot(S.T, S)))
	   	M[i] = np.sqrt(max_nu) * S
	   	active_set.append(ni)
	   	R = R.drop([ni])

if __name__ == '__main__':
	"""The function is invoked using dummy values"""
	size = 1000
    X, _ = datasets.make_regression(n_samples = 10000, n_features = 10, n_targets = 1)
    kernel = GPy.kern.RBF(1)
    beta = 0.5
	ivm_regression(size = size, data = X, kernel = kernel, beta = beta)