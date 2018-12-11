############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: LU

# Citation: 
# PEREIRA, V. (2018). Project: LU, File: Python-DM-LUPD.py, GitHub repository: <https://github.com/Valdecy/LU_Decomposition>

############################################################################

# Required Library
import numpy as np

# Function
def LU_partial_decomposition(matrix):
    n, m = matrix.shape
    P    = np.identity(n)
    L    = np.identity(n)
    U    = matrix.copy()
    PF   = np.identity(n)
    LF   = np.zeros((n,n))
    for k in range(0, n - 1):
        index = np.argmax(abs(U[k:,k]))
        index = index + k 
        if index != k:
            P = np.identity(n)
            P[[index,k],k:n] = P[[k,index],k:n]
            U[[index,k],k:n] = U[[k,index],k:n] 
            PF = np.dot(P,PF)
            LF = np.dot(P,LF)
        L = np.identity(n)
        for j in range(k+1,n):
            L[j,k]  = -(U[j,k] / U[k,k])
            LF[j,k] =  (U[j,k] / U[k,k])
        U = np.dot(L,U)
    np.fill_diagonal(LF, 1)
    return PF, LF, U

# Usage
A = [[2, 1, 1, 0], [4, 3, 3, 1], [8, 7, 9, 5], [6, 7, 9, 8]]
A = np.array(A)
P1, L1, U1 = LU_partial_decomposition(A)
