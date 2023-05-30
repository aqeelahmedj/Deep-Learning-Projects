# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:50:35 2023

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)

# Generate data
x = np.linspace(-10, 10, 100)
y = leaky_relu(x)

# Plotting
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('Leaky ReLU(x)')
plt.title('Leaky ReLU Activation Function')
plt.grid(True)
plt.show()
