import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtracting the maximum value for numerical stability
    return e_x / np.sum(e_x, axis=0)

# Example usage
x = np.array([2.0, 1.0, 0.5])
output = softmax(x)
print(output)
