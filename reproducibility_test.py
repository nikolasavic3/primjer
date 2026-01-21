# Simple ML Reproducibility Test
# Eigenvalue decomposition WILL differ between BLAS implementations!

import numpy as np

np.random.seed(42)

# Create a symmetric matrix (for eigenvalue decomposition)
A = np.random.rand(50, 50)
A = A @ A.T  # Make symmetric

# Eigenvalue decomposition - HIGHLY BLAS dependent!
eigenvalues, eigenvectors = np.linalg.eig(A)

# Sort eigenvalues for consistent comparison
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]

# These WILL differ between Accelerate and OpenBLAS
print(f"NumPy version: {np.__version__}")
print(f"Top 5 eigenvalues: {eigenvalues[:5]}")
print(f"Eigenvalue sum: {eigenvalues.sum():.15f}")
print(f"Eigenvector hash: {hash(eigenvectors.tobytes())}")

# Show BLAS info
print(f"\nBLAS config:")
np.show_config()
