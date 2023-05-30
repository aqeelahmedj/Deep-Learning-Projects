import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

# Generate data
x = np.linspace(-10, 10, 100)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)
y_tanh = tanh(x)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].plot(x, y_sigmoid, color='blue')
axs[0, 0].set_title('Sigmoid')

axs[0, 1].plot(x, y_relu, color='red')
axs[0, 1].set_title('ReLU')

axs[1, 0].plot(x, y_leaky_relu, color='green')
axs[1, 0].set_title('Leaky ReLU')

axs[1, 1].plot(x, y_tanh, color='purple')
axs[1, 1].set_title('Tanh')

for ax in axs.flat:
    ax.set(xlabel='x', ylabel='Activation Value')
    ax.grid(True)

plt.tight_layout()
plt.show()
