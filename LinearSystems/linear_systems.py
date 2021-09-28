# linear_systems.py
"""Volume 1: Linear Systems.
Bryant McArthur
Math 345 Sec 002
September 28
"""

import numpy as np
import time
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla
from matplotlib import pyplot as plt


def first_column_zeros(A):
  B=np.copy(A)
  (m,n)=B.shape
  for i in range(m-1):
    B[i+1,:]=B[i+1,:]-(B[i+1,0]/B[0,0])*B[0,:]
  return B

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    
    B=np.copy(A)
    (m,n)=B.shape
    
    #For every row call the function to make the columns zeros below the diagonal.
    for i in range(n-1):
      C=np.copy(B)
      C=first_column_zeros(C[i:,i:])
      B[i:,i:]=C
      
    return B
        
  

A = np.random.randint(1, high=12, size=(3,3))
#print(A)
#print(ref(A))

# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    U=np.copy(A)
    L=np.identity(len(A))
    (m,n)=A.shape
    for j in range(n-1):
      if U[j,j]==0:
          U[j,:]=A[j+1,:]
          U[j+1,:]=A[j,:]
      for i in range(j+1,m):
        L[i,j]=U[i,j]/U[j,j]
        U[i,:]=U[i,:]-L[i,j]*U[j,:]
        
    return L,U 


#print(A)
#print(lu(A))


def forward_substitution(L,b): # Accepts a lower triangular square matrix L and a vector b, solves Ly=b for y.
  n=len(b)
  y = []
  for i in range(n):
    y.append(0)
  for i in range(n):
    y[i]=(b[i]-np.dot(y,L[i,:]))/L[i,i]
  y=np.array(y)
  return y



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

# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    
    L,U = lu(A)
    
    y = forward_substitution(L,b)
    x = back_substitution(U, y)
    
    return x
    


# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    # Create lists
    n_times = [2**n for n in range(1,9)]
    invtimes = [None] * 8
    lasolvetimes = [None] * 8
    factortimes = [None] * 8
    lalusolvetimes = [None] * 8
    
    for i in range(1,len(n_times)):
        A = np.random.random((n_times[i],n_times[i]))
        b = np.random.random(n_times[i])
        
        #1
        start = time.perf_counter()
        Ainv = la.inv(A)
        x = Ainv @ b
        invtimes[i] = time.perf_counter() - start
        
        #2
        start = time.perf_counter()
        la.solve(A,b)
        lasolvetimes[i] = time.perf_counter() - start
        
        #3
        start = time.perf_counter()
        L,P = la.lu_factor(A)
        x = la.lu_solve((L,P),b)
        factortimes[i] = time.perf_counter() - start
        
        #4
        L,P = la.lu_factor(A)
        start = time.perf_counter()
        x = la.lu_solve((L,P),b)
        lalusolvetimes[i] = time.perf_counter() - start
    
    
    #Linear plot
    ax1 = plt.subplot(121)
    ax1.plot(n_times,invtimes, 'b.-', linewidth=1.5, markersize=10, label="Using la.inv()")
    ax1.plot(n_times,lasolvetimes, '.-',color="orange", linewidth=1.5, markersize=10, label="Using la.solve()")
    ax1.plot(n_times,factortimes, 'k.-', linewidth=1.5, markersize=10, label="LU including factoring")
    ax1.plot(n_times,lalusolvetimes, '.-',color="yellow", linewidth=1.5, markersize=10, label="LU just solving")
    ax1.legend(loc="upper left")
    plt.xlabel("n",fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    
    
    #Logrithmic Plot
    ax2 = plt.subplot(122)
    ax2.loglog(n_times,invtimes, 'b.-', linewidth=1.5, markersize=10)
    ax2.loglog(n_times,lasolvetimes, '.-',color="orange", linewidth=1.5, markersize=10)
    ax2.plot(n_times,factortimes, 'k.-', linewidth=1.5, markersize=10)
    ax2.plot(n_times,lalusolvetimes, '.-',color="yellow", linewidth=1.5, markersize=10)
    plt.xlabel("n",fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    
    ax1.set_title("Linear", fontsize=8)
    ax2.set_title("Logrithimic", fontsize=8)
    
    plt.suptitle("Time taken to solve Ax = b")
    
    plt.tight_layout()
    plt.show()
    
    
#prob4()

# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    
    diagonals = [1,1,-4,1,1]
    offsets = [-2,-1,0,1,2]
    
    A = sparse.diags(diagonals, offsets, shape=(n**2,n**2))
    """
    plt.spy(A, markersize=1)
    plt.show()
    """
    return A

prob5(10)



# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    
    n_times = [2**n for n in range(1,8)]
    CSRtimes = [None] * 7
    NumPytimes = [None] * 7
    
    for i in range(len(n_times)-1):
        A = prob5(n_times[i])
        b = np.random.random(n_times[i]*n_times[i])
        
        #Converting A to CSR Format
        Acsr = A.tocsr()
        start = time.perf_counter()
        x = spla.spsolve(Acsr,b)
        CSRtimes[i] = time.perf_counter() - start
        
        #Converting A to NumPy array
        Anumpy = A.toarray()
        start = time.perf_counter()
        la.solve(Anumpy,b)
        NumPytimes[i] = time.perf_counter() - start
        
        
    #Linear plot
    ax1 = plt.subplot(121)
    ax1.plot(n_times,CSRtimes, 'b.-', linewidth=1.5, markersize=10, label="CSR Format")
    ax1.plot(n_times,NumPytimes, '.-',color="orange", linewidth=1.5, markersize=10, label="NumPy Array")        
    ax1.legend(loc="upper left")
    plt.xlabel("n",fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
        
    #Logrithmic Plot
    ax2 = plt.subplot(122)
    ax2.loglog(n_times,CSRtimes, 'b.-', linewidth=1.5, markersize=10)
    ax2.loglog(n_times,NumPytimes, '.-',color="orange", linewidth=1.5, markersize=10)
    plt.xlabel("n",fontsize=14)
    
    ax1.set_title("Linear", fontsize=8)
    ax2.set_title("Logrithimic", fontsize=8)
    
    plt.suptitle("Time taken to solve Ax=b for sparse matrix")
    
    plt.tight_layout()
    plt.show()

prob6()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

