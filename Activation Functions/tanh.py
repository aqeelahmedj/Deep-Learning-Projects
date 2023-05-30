# -*- coding: utf-8 -*-
"""
Created on Tue May 30 10:51:23 2023

@author: DELL
"""

import numpy as np
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

# Generate data
x = np.linspace(-10, 10, 100)
y = tanh(x)

# Plotting
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.title('Tanh Activation Function')
plt.grid(True)
plt.show()
