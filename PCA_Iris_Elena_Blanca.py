# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 15:05:23 2016

@author: BlancAlee
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from sklearn import datasets

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
iris = datasets.load_iris()
X = iris.data
y = iris.target

X1 = X[:,0].mean() #Obtain the mean from the first column 
Xa = np.subtract(X[:,0], X1) #Subtrancts the mean from the first column
X2 = X[:,1].mean() #Obtain the mean from the second column 
Xb = np.subtract(X[:,1], X2) #Subtrancts the mean from the second column
X3 = X[:,2].mean() #Obtain the mean from the third column 
Xc = np.subtract(X[:,2], X3) #Subtrancts the mean from the third column
X4 = X[:,3].mean() #Obtain the mean from the fourth column 
Xd = np.subtract(X[:,3], X4) #Subtrancts the mean from the fourth column

"""adding all the columns to form a matrix, and transposing the matrix"""
DatasetN = np.column_stack((Xa, Xb, Xc, Xd)) 
DatasetNtrans = np.transpose(DatasetN)

covMtx = np.dot(DatasetNtrans, DatasetN) #Obtain the covariance matrix 


w, v = LA.eig(covMtx) #Obtains both the eigenvalues and eigenvectors

"""delete the smallest of both the values and the vectors"""
w1 = np.delete(w, 3)
v1 = np.delete(v, 3, 1)

"""Obtain the transpose from the vectors and from x"""
v_transpose = np.transpose(v1)
x_transpose = np.transpose(X)

"""Obtain the new data set"""
NewDataS = np.dot(v_transpose, x_transpose)
n_trans = np.transpose(NewDataS)

"""Ploting the 3 D graph"""
fig = plt.figure(1, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(n_trans[y == label, 0].mean(),
              n_trans[y == label, 1].mean() + 1.5,
              n_trans[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(n_trans[:, 0], n_trans[:, 1], n_trans[:, 2], c=y, cmap=plt.cm.spectral)

x_surf = [n_trans[:, 0].min(), n_trans[:, 0].max(),
          n_trans[:, 0].min(), n_trans[:, 0].max()]
y_surf = [n_trans[:, 0].max(), n_trans[:, 0].max(),
          n_trans[:, 0].min(), n_trans[:, 0].min()]
x_surf = np.array(x_surf)
y_surf = np.array(y_surf)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()