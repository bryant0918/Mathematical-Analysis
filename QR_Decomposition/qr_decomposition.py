# qr_decomposition.py
"""Volume 1: The QR Decomposition.
Bryant McArthur
Math 345
Sec 002
"""

import numpy as np
from scipy import linalg as la


# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    
    m,n = A.shape
    Q = A.copy()
    R = np.zeros((n,n))
    
    #Normalize the ith column of Q
    for i in range(n):
        R[i,i] = la.norm(Q[:,i])
        Q[:,i] = Q[:,i]/R[i,i]
        
        #Orthogonalize the jth column of Q
        for j in range(i+1,n):
            R[i,j] = Q[:,j].T @ Q[:,i]
            Q[:,j] = Q[:,j] - R[i,j] * Q[:,i]
            
    return Q,R
    
A = np.random.randint(-10,10,size=(6,4))
Q,R = qr_gram_schmidt(A)


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    Q,R = la.qr(A, mode="economic")
    diags = np.diag(R)
    B = np.prod(diags)
    
    return abs(B)


#Problem 3

def back_substitution(U,y):    # Accepts an upper triangular square matrix U and a vector b, solves Ux=b for x.
      n=len(y)
      x=[]
      for i in range(n):
        x.append(0)
      for i in range(n):
        j=abs(i-n+1)
        x[j]=(y[j]-np.dot(x,U[j,:]))/U[j,j]
      x=np.array(x)
      return x
  

def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    
    Q,R = la.qr(A, mode="economic")
    y = Q.T @ b
    x = back_substitution(R,y)

    return x

# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    
    m,n = A.shape
    R = A.copy()
    Q = np.identity(m)
    sign = lambda x: 1 if x >= 0 else -1
    
    #Normalize u, and Reflect R and Q
    for k in range(n):
        u = R[k:,k].copy()
        u[0] = u[0] + sign(u[0]) * la.norm(u)
        u = u/la.norm(u)
        R[k:,k:] = R[k:,k:] - 2*np.outer(u, (u.T @ R[k:,k:]))
        Q[k:,:] = Q[k:,:] - 2*np.outer(u, (u.T @ Q[k:,:]))
        
    return Q.T, R

# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    
    m,n = A.shape
    H = A.copy()
    Q = np.identity(m)
    
    #Apply Qk to H, Qk.T to H and Qk to Q
    for k in range(n-2):
        u = H[k+1:,k].copy()
        u[0] = u[0] + np.sign(u[0]) * la.norm(u)
        u = u / la.norm(u)
        H[k+1:,k:] = H[k+1:,k:] - 2 * np.outer(u, (u.T @ H[k+1:,k:]))
        H[:,k+1:] = H[:,k+1:] - 2 * np.outer((H[:,k+1:] @ u), u.T)
        Q[k+1:,:] = Q[k+1:,:] - 2 * np.outer(u, (u.T @ Q[k+1:,:]))
        
    return H,Q.T

A = np.random.random((8,8))
H,Q = hessenberg(A)

