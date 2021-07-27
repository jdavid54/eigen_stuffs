import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

# https://secure.math.ubc.ca/~pwalls/math-python/linear-algebra/eigenvalues-eigenvectors/

'''
Let A be a square matrix. A non-zero vector v is an eigenvector for A with eigenvalue k if:

    A.v = k.v

Rearranging the equation, we see that v is a solution of the homogeneous system of equations :
    
    (A - kI)v = 0         where  is the identity matrix of size n.

Non-trivial solutions exist only if the matrix (A - kI) is singular which means det(A - kI) = 0.
Therefore eigenvalues of A are roots of the characteristic polynomial :
    
    p(k) = det(A-kI)

'''
A = np.array([[1,0],[0,-2]])
print(A)

I = np.eye(2,2)

results = la.eig(A)
print(results[0])
print(results[1])


eigvals, eigvecs = la.eig(A)
print(eigvals)
print(eigvecs)
print(eigvals*I, A)


eigvals = eigvals.real
print('eigvals :',eigvals)

lambda1 = eigvals[1]
print('lambda1 :', lambda1)

v1 = eigvecs[:,1].reshape(2,1)
print(v1)

print(A @ v1)
print(lambda1 * v1)

# Symmetric Matrices
print('Symmetric Matrices')
n = 4
#P = np.random.randint(0,10,(n,n))
P = np.array(
    [[7, 0, 6, 2],
     [9, 5, 1, 3],
     [0, 2, 2, 5],
     [6, 8, 8, 6]])
print(P)
evals, evecs = la.eig(P)
print('Eigen stuff for P\n', evals, evecs)

v1 = evecs[:,0].reshape(n,1) # reshape first vector
print('v1 :',v1)
k1 = evals[0]
print('val1 :', k1)

# P @ v1 = k1 * v1
print('P @ v1 :', P @ v1)
print('k1 * v1 :', k1 * v1)

print('==========================================')
# @ = matrix multiplication
S = P @ P.T
print('S :', S)

evals, evecs = la.eig(S)
print(evals)

evals = evals.real
print('eigen real values \n', evals)
print('eigen vectors\n', evecs)  # vectors in columns

v1 = evecs[:,0] # First column is the first eigenvector
print(v1)

v2 = evecs[:,1] # Second column is the second eigenvector
print(v2)

v3 = evecs[:,2] # Third column is the third eigenvector
print(v3)

print(v1 @ v2)  # zero orthogonal vectors
print(v1 @ v3)



#Diagonalization
print('==========================================')
print('Diagonalization')
'''
A square matrix M is diagonalizable if it is similar to a diagonal matrix.
In other words, M is diagonalizable if there exists an invertible matrix P
such that D = P(-1).M.P  is a diagonal matrix with  P(-1) = inverse of P.

A beautiful result in linear algebra is that a square matrix M of size
is diagonalizable if and only if M has n independent eigevectors.
Furthermore, M = PDP(-1) where the columns of P are the eigenvectors of M and D
has corresponding eigenvalues along the diagonal.
'''
# eigen vectors
P = np.array([[1,1],[1,-1]])
print('P :', P)

print(np.eye(2,2)*[3,1]) # diagonal matrix D
# diagonal with eigen values in diagonal
# D = k*I 
D = np.diag((3,1))   # error with power 20 because dtype = int32
# use uint32, float or int64
D = np.array(np.diag((3,1)), dtype = np.uint32)
print(D)

M = P @ D @ la.inv(P)
print(M)

evals, evecs = la.eig(M)
print(evals)
print(evecs)

# Matrix Powers

Pinv = la.inv(P)
k = 20

# https://docs.python.org/3/library/timeit.html
# timing : timeit.timeit
import timeit
result = M.copy()
start = timeit.timeit()
for _ in range(1,k):
    result = result @ M
end = timeit.timeit()
print('timeit', result, end - start)

# timing : timeit.default_timer
from timeit import default_timer as timer
result1 = M.copy()
start = timer()
for _ in range(1,k):
    result1 = result1 @ M
end = timer()
t1 = end - start
print('k power M', result1, t1) # Time in seconds, e.g. 5.38091952400282


start = timer()
result2 = P @ D**k @ Pinv
end = timer()
t2 = end - start
print('diagonalisation', result2, t2)

print(t2<t1, t1/t2)