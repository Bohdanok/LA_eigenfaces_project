import numpy as np

def power_iteration(A, k = 100):
    if k < 1:
        raise ValueError("k should be >= 1")
    # start with random vector to reduce probability of getting a vector
    # which is orthogonal to most dominant eigenvector
    b_k = np.random.rand(A.shape[1], 1)
    eigenvalue = 0
    for i in range(k):
        b_k_new = np.matmul(A, b_k)
        b_k_new_norm = np.linalg.norm(b_k_new)
        # normalize the vector
        b_k = b_k_new / b_k_new_norm
        eigenvalue = b_k_new_norm
    eigenvector = b_k
    return eigenvector, eigenvalue


def svd(A, num_iter=100):
    A = A.astype(float).copy()
    m, n = A.shape
    ATA = A.T @ A
    eigenvalues, eigenvectors = [], []
    # find all eigenvectors and eigenvalues
    for _ in range(n):
        v, eigenvalue = power_iteration(ATA, num_iter)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(v)
        # get matrix without previous eigenvectors effect of previous eigenvectors
        ATA -= eigenvalue * np.outer(v, v)
    eigenvalues = np.array(eigenvalues)
    sigma = np.sqrt(np.maximum(eigenvalues, 0))
    V = np.column_stack(eigenvectors)
    U = np.zeros((m, m))
    tolerance = 1e-6
    for i in range(n):
        if sigma[i] > tolerance:
            U[:, i] = (A @ V[:, i]) / sigma[i]
    return U, sigma, V.T
