# Simple ML Reproducibility Test
# Matrix operations can give DIFFERENT results on Mac vs Linux!

import numpy as np

np.random.seed(42)

# Create matrices
A = np.random.rand(100, 100)
B = np.random.rand(100, 100)

# Matrix multiplication - uses BLAS (Accelerate on Mac, OpenBLAS on Linux)
C = A @ B

# These values may DIFFER slightly between Mac and Linux!
print(f"NumPy version: {np.__version__}")
print(f"Matrix result checksum: {C.sum():.15f}")
print(f"Matrix [0,0] element:   {C[0,0]:.15f}")

# Save for comparison
with open("result.txt", "w") as f:
    f.write(f"{C.sum():.15f}\n")
