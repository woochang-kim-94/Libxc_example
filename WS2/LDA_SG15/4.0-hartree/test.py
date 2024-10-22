import numpy as np

a = np.array([1, 2, 3, 4, 4])
b = np.array([1, 3, 4, 0, 5])

# Create a copy of array b to avoid modifying the original array
c = np.copy(b)

# Find indices where b is zero
zero_indices = np.where(b == 0)

# Perform element-wise division for non-zero elements
c[~zero_indices] = a[~zero_indices] / b[~zero_indices]

# Set the elements to zero where b is zero
c[zero_indices] = 0

print(c)
