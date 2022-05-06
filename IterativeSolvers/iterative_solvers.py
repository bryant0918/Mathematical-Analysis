# iterative_solvers.py
"""Volume 1: Iterative Solvers.
Bryant McArthur
Sec 002
March 8
"""

import numpy as np
from scipy import linalg as la
from scipy import sparse
from matplotlib import pyplot as plt


# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    #Initialize X0 and get the diagonals of A
    n = len(A)
    x0 = np.zeros(n)
    diagonal = np.diag(A)
    
    #Get the inverse
    inv = np.ones(n)/diagonal
    Dinv = np.diag(inv)
    
    #Make first iteration
    xk = x0 + Dinv@(b-A@x0)
    
    #Initialize list of first error
    k = 1
    errors = [la.norm(A@xk-b, np.inf)]
    
    #Iterate until converges or we reach maxiterations
    while la.norm(x0 - xk, ord = np.inf) >= tol and k < maxiter:
        #Find xk and error
        x0 = xk
        xk = x0 + Dinv@(b-A@x0)
        k += 1
        errors.append(la.norm(A@xk-b, np.inf))
        
    #Plot error if wanted
    if plot:
        plt.semilogy(range(k), errors)
        plt.title("Convergence of Jacobi Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error")
        
        
    return xk
        
        
b = np.random.random(8)
#print(b)
A = diag_dom(8)
print(jacobi(A,b,plot=True))


# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    #Initialize everything
    x0 = np.zeros(len(A))
    errors = []
    #Iterate until converges or we reach max iterations
    for k in range(maxiter):
        #Make a copy because its mutable
        xk = np.copy(x0)
        
        #Find next xk
        for i in range(len(A)):
            xk[i] = xk[i] +(1/A[i,i])*(b[i] - np.dot(A[i,:], xk))
        
        if np.linalg.norm(xk - x0, np.inf) <= tol:
            break
        
        errors.append(np.linalg.norm(A@xk-b, np.inf))
        
        x0 = xk
    #print(errors)
        
    #Plot if wanted    
    if plot:
        plt.semilogy(range(k), errors)
        plt.title("Convergence of Gauss_seidel Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error")
        plt.show()
        
        
    #print(k)
    
    return xk

#print(gauss_seidel(A,b,plot=True))
A = diag_dom(100)
b = np.random.random(100)
x = gauss_seidel(A, b, plot=True)
#print(np.allclose(A@x,b))

# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    #Initialize x0 and diagonals
    x0 = np.zeros(len(b))
    diag = A.diagonal()
    
    #Iterate at most maxiter times
    for k in range(maxiter):
        #Make a copy
        xk = x0.copy()
        #Iterate through columns of A
        for i in range(len(b)):
            #Use the handy functions from the file
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            Aix = A.data[rowstart:rowend] @ xk[A.indices[rowstart:rowend]]
            
            #Find xk
            xk[i] = x0[i] + (1/diag[i]) * (b[i] - Aix)
            
        #Break if converges
        if la.norm(x0-xk) < tol:
            break
        
        #Reset
        x0 = xk
    
    return xk

#A = sparse.csr_matrix(diag_dom(5000))
#b = np.random.random(5000)
#print(gauss_seidel_sparse(A, b))


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    #Initialize everything
    x0 = np.zeros(len(b))
    diag = A.diagonal()
    converged = False
    
    #Iterate at most max iter times
    for k in range(maxiter):
        #Make a copy
        xk = x0.copy()
        
        #Go through columns of A
        for i in range(len(b)):
            #Use function from book
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            Aix = A.data[rowstart:rowend] @ xk[A.indices[rowstart:rowend]]
            
            #Find xk
            xk[i] = x0[i] + (omega/diag[i]) * (b[i] - Aix)
            
        #If converges break and return converged = True
        if la.norm(x0-xk) < tol:
            converged = True
            break
        
        #Reset
        x0 = xk
    
    return xk, converged, k+1


# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    #Find A and B from linear systems lab
    diagonals = [1, -4, 1]
    offdiags = [-1, 0, 1]
    B = sparse.diags(diagonals, offdiags, shape = (n,n))
    A = sparse.block_diag([B]*n)
    A.setdiag(1, -n)
    A.setdiag(1, n)
    
    #Use tile function to find b
    b = np.tile(np.concatenate(([-100], [0]*(n-2), [-100])), n)
    
    #Call problem 5 to get itervalue
    soln, conv, k = sor(A.tocsr(), b, omega=omega, tol=tol, maxiter=maxiter)
    
    #Plot if wanted
    if plot:
        arr = soln.reshape((n,n))
        plt.pcolormesh(arr)
        plt.title("Heat on the Plate")
        plt.show()
        
    return soln, conv, k
    
    



# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    #Make linspace from 1 to 1.95
    omegas = np.linspace(1,1.95,20)
    iters = []
    
    #Go through values in linspace
    for omega in omegas:
        #Call problem 6 to get iteration
        soln, conv, k = hot_plate(20, omega, tol=1e-2, maxiter = 1000)
        iters.append(k)
        
    #Plot even if not wanted
    plt.plot(omegas, iters)
    plt.title("Iterations vs. Omega Value")
    plt.xlabel("Omega")
    plt.ylabel("Iteration")
    plt.show()
    
    return omegas[np.argmin(iters)]

#print(prob7())