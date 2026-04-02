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
                        
    
    if np.array_equal(H, H.conj().T):     # Hermitian check
        return H
    else:
        return "Not Hermitian"

def plot_eigens(H, N):
    eigenvalues, eigenvectors = np.linalg.eigh(H)

    plt.figure(1)
    for k in range(3):
        plt.plot(range(N), eigenvectors[:,k]**2, label=f"State {k}")    # eigenvectors**2 vs. site #
    
    plt.legend()
    plt.xlabel("Site #")
    plt.ylabel("Probability")
    #plt.show()
    
    return eigenvalues, eigenvectors
    
def DOS(evals, gamma):
    E_grid = np.linspace(evals.min() - 1, evals.max() + 1, num=500)
    dos = np.zeros_like(E_grid)
    
    for En in evals:
        dos += Lorentzian(E_grid, En, gamma)
    dos /= len(evals)   # Normalize
    
    plt.figure(2)
    plt.plot(E_grid, dos, label="DOS")    # Plotting DOS
    plt.hist(evals, label="Eigenvalues", density=True)    # Eigenvaluess histogram normalized
    #plt.plot(range(N), evals, label="Eigenvalues")
    
    
    plt.legend()
    plt.xlabel("Energy")
    plt.ylabel("Eigenvalue Density")
    plt.title("DOS")
    #plt.show()
    
    return dos
    

def Lorentzian(E, En, gamma):   # Approximation of Delta Function
    return ((1 / np.pi) * (1/2 * gamma) / ((E - En)**2 + (1/2 * gamma)**2))    # Lorentzian centered at En

def LDOS(evals, eigenvects, gamma, N):
    E_grid = np.linspace(evals.min() - 1, evals.max() + 1, num=500)
    ldos = np.zeros_like(E_grid)
    sites = np.linspace(0, N - 1, 7, dtype=int)    # Some evenly-space list of sites
    
    plt.figure(3)
    for i in range(len(sites)):
        for n in range(len(evals)):
            ldos += (eigenvects[sites[i], n]**2) * Lorentzian(E_grid, evals[n], gamma)    # Psi^2 * delta
        plt.plot(E_grid, ldos, label=f"site {sites[i] + 1}")
        ldos = np.zeros_like(E_grid)
    
    plt.legend()
    plt.xlabel("Energy")
    plt.ylabel("Density of States")
    plt.title("LDOS")
    #plt.show()
    
    return ldos

# Variables
N = 1000
t = 1   
condition = "open"  # "periodic" or "open"
nn = 1      # 0 for no jumping
eps = np.full(N, 0)
gamma = 0.1


H = H_constcut(N, t, eps, condition, nn)

eigenvalues, eigenvectors, = plot_eigens(H, N)

dos = DOS(eigenvalues, gamma)

ldos = LDOS(eigenvalues, eigenvectors, gamma, N)

print(H)

plt.figure(4)
plt.plot(range(N), eigenvalues, label="Eigenvalues")
plt.show()

