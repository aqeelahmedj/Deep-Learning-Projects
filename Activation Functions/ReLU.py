# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:46:13 2023

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# Generate data
x = np.linspace(-10, 10, 100)
y = relu(x)

# Plotting
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.title('ReLU Activation Function')
plt.grid(True)
plt.show()


