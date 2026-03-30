import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def H_constcut(N, t, eps, condition, nn):
    H = np.zeros((N, N))
    
    for i in range(N):
        H[i, i] = eps[i]
        for j in range(1, nn + 1):
            if (i - j) >= 0:
                H[i, i - j] = t / j
            elif condition == "periodic":   # only wrap if out of bounds
                H[i, (N + i) - j] = t / j

            if (i + j) < N:
                H[i, i + j] = t / j
            elif condition == "periodic":   # only wrap if out of bounds
                H[i, (i + j) - N] = t / j
                        
    print(f"nn={nn}")
    print(f"Is Hermitian: {np.allclose(H, H.T)}")
    print(f"Diagonal (first 5): {np.diag(H)[:5]}")
    print(f"Row 0 nonzero elements: {np.nonzero(H[0])}")
    print(f"Row 1 nonzero elements: {np.nonzero(H[1])}")
    print(f"Row 2 nonzero elements: {np.nonzero(H[2])}")
    print(f"Max off-diagonal value: {np.max(np.abs(H - np.diag(np.diag(H))))}")
    
    if np.array_equal(H, H.conj().T):     # Hermitian check
        return H
    else:
        return "Not Hermitian"

def plot_eigens(H, N):
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    for k in range(3):
        plt.plot(range(N), eigenvectors[:,k]**2, label=f"State {k}")    # eigenvectors**2 vs. site #
    
    plt.legend()
    plt.xlabel("Site #")
    plt.ylabel("Probability")
    plt.show()
    
    return eigenvalues, eigenvectors
    
def DOS(evals, gamma):
    x_vals = np.linspace(evals.min(), evals.max(), num=100)
    
    dos = np.zeros_like(x_vals)
    for e in evals:
        dos += Lorentzian(x_vals, gamma, e)
    dos /= len(evals)   # Normalize
    
    plt.plot(x_vals, dos, label="DOS")    # Plotting DOS
    plt.hist(evals, label="Eigenvalues", density=True)    # Eigenvaluess histogram normalized
    #plt.plot(range(N), evals, label="Eigenvalues")
    
    
    plt.legend()
    plt.xlabel("Energy")
    plt.ylabel("Eigenvalue Density")
    plt.title("DOS")
    plt.show()
    
    return dos
    

def Lorentzian(x, gamma, energy):   # Approximation of Delta Function
    return ((1 / np.pi) * (1/2 * gamma) / ((energy - x)**2 + (1/2 * gamma)**2))    # Lorentzian centered at energy

def LDOS(evals, eigenvects, gamma, site):
    x_vals = np.linspace(evals.min(), evals.max(), 500)
    ldos = np.zeros_like(x_vals)
    
    for n in range(len(evals)):
        ldos += (eigenvects[site, n]**2) * Lorentzian(x_vals, gamma, evals[n])
    plt.plot(x_vals, ldos)
    
    plt.xlabel("Energy")
    plt.ylabel("Density of States")
    plt.title("LDOS")
    plt.show()
    
    return ldos

# Variables
N = 500
t = 1   
condition = "open"  # "periodic" or "open"
nn = 1      # 0 for no jumping
eps = np.full(N, 0)
gamma = 0.25

#print(eps)
#print(type(eps))
H = H_constcut(N, t, eps, condition, nn)
#print(eps)
#print(type(eps))
eigenvalues, eigenvectors, = plot_eigens(H, N)

dos = DOS(eigenvalues, gamma)

ldos = LDOS(eigenvalues, eigenvectors, gamma, 1)

print(H)



