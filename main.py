#!/usr/bin/env python
# coding: utf-8

# Librerias
import numpy as np
import matplotlib.pyplot as ptl

# Libreria con el dataset de casas de boston
from sklearn.datasets import load_boston

# Info del dataset
boston = load_boston()
print(boston.DESCR)

x = np.array(boston.data[:, 5])
y = np.array(boston.target)

# Mostrar datos de las casas de Boston en una grafica
ptl.scatter(x, y, alpha=0.3)

# se agrega columna de unos para los terminos independientes
x = np.array([np.ones(506), x]).T


"""
Ecuacion de una recta: y = mx + b
En una matriz seria: Y = A*U
donde Y y A son matrices y U tiene las variables b y m => [b,m]
como Y - AU no puede ser cero (0) se busca una matriz U con el minimo de Y-AU
que seria: U = (((X^t)*X)^-1)*(X^t)Y 

@ => Multiplica matrices
.T => Tranpuesta de una matriz
"""
# Formula para minimizar el error cuadratico medio
B = np.linalg.inv(x.T @ x) @ x.T @ y 

# Mostrar la regresion lineal en la grafica
# ptl.plot([x,x1], [y, y1])
ptl.plot([4,9], [B[0] + B[1] * 4, B[0] + B[1] * 9], c="red")
ptl.show()



