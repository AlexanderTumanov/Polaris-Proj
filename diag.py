import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def H_constcut(N, t, eps, condition, nn):
    H = np.zeros((N, N))
    
    for i in range(N):
        H[i, i] = eps[i]
        for j in range (nn + 1):
            if j > 0:
                if (i - j) >= 0:     # Non-periodic left-nn
                    H[i, i - j] = t
                if (i + j) < N:    # Non-periodic right nn
                    H[i, i + j] = t
                if condition == "periodic":
                    if (i - j) < 0:    # Periodic left nn
                        H[i, (N + i) - j] = t
                    if (i + j) >= N:   # Periodic right nn
                        H[i, (i + j) - N] = t
                        
                    
    if np.array_equal(H, H.conj().T):     # Hermitian check
        return H
    else:
        return "Not Hermitian"

def plot_eigens(H, N):
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    for k in range(4):
        plt.plot(range(N), eigenvectors[:,k]**2, label=f"State {k}")    # eigenvectors**2 vs. site #
    
    plt.legend()
    plt.xlabel("Site #")
    plt.ylabel("Probability")
    plt.show()
    
    return eigenvalues, eigenvectors
     
    
N = 50
t = 1   
condition = "periodic"
nn = 1      # 0 for no jumping
eps = np.full(N, 0)
gamma = 0.1
energy = 0
H = H_constcut(N, t, eps, condition, nn)
eigenvalues, eigenvectors, = plot_eigens(H, N)


