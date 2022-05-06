# condition_stability.py
"""Volume 1: Conditioning and Stability.
Bryant McArthur
Sec 002
February 1, 2022
"""

import numpy as np
import sympy as sy
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    #Get the singular values
    U,S,_ = la.svd(A)
    sn = S[0]
    s1 = S[len(S)-1]
    
    #Return infinity if smallest singular value is 0
    if s1 == 0:
        return np.inf
    
    #Find K
    K = sn/s1
    
    return K


A = np.array([[1,2,3],[4,5,6],[7,8,9]])
#print(matrix_cond(A))



# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    
    w_roots = np.arange(1, 21)
    

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())
    
    roots, absks, relks = [],[],[]
    
    #Iterate 100 times
    for i in range(100):
        #Draw from normal distribution for r value
        r = np.random.normal(1,10e-10,21)
        d = la.norm(r,np.inf)
        
        #Perturb coefficients by r
        new_coeffs = w_coeffs*r
        
        #Find the new roots
        new_roots = np.roots(np.poly1d(new_coeffs))
        
        #Sort the roots
        w_roots = np.sort(w_roots)
        new_roots = np.sort(new_roots)
        
        #Plot the real and imaginary parts
        plt.scatter(np.real(new_roots), np.imag(new_roots), s=1, marker = '.',color = 'k',)
        
        #Calculate the condition numbers
        absk = la.norm(new_roots - w_roots, np.inf) / d
        absks.append(absk)
        
        relk = absk * la.norm(w_coeffs,np.inf) / la.norm(w_roots, np.inf)
        relks.append(relk)
    
    #Plot it
    plt.scatter(np.real(w_roots),np.imag(w_roots))
    plt.title("Roots of perturbed Wilkinson Polynomial on Complex plane")
    plt.show()
        
    #Find the average absolute and relative condition numbers
    avgabsk = np.mean(absks)
    avgrelk = np.mean(relks)
    
    return avgabsk, avgrelk

print(prob2())


# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
        
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    n = len(pert_eigvals)
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1,1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0],:] = np.inf
        dists[:,index[1]] = np.inf
    return pert_eigvals[sort_order]

# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    #Initialize everything
    reals = np.random.normal(0,1e-10,A.shape)
    imags = np.random.normal(0,1e-10,A.shape)
    H = reals + 1j*imags
    
    #Find the eigenvalues of A and A+H
    l = la.eigvals(A,homogeneous_eigvals=False)
    ltilda = la.eigvals(A+H,homogeneous_eigvals=False)
    
    #Reorder with handy function
    ltilda = reorder_eigvals(l,ltilda)
    
    #Find the absolute condition number
    Khat = la.norm(l-ltilda,2)/la.norm(H,2)
    
    #Find the relative condition number
    K = la.norm(A,2)/la.norm(l,2)*Khat
    
    return Khat, K


# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    #Initialize everything
    Ks = np.ones((res,res))
    x = np.linspace(domain[0],domain[1],res)
    y = np.linspace(domain[2],domain[3],res)
    
    #Iterate through x and y values
    for i in range(len(x)-1):
        for j in range(len(y)-1):
            #Use problem 3 to find K value
            K = eig_cond(np.array([[1,x[i]],[y[j],1]]))[1]
            Ks[i][j] = K
            
    #Plot it
    plt.pcolormesh(x,y,Ks,cmap='gray_r')
    plt.colorbar()
    plt.title("Eigenvalue Problem")
    plt.show()
    
    return

#prob4(res=200)

# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    #Load the data and initialize A,x and b
    x,b = np.load("stability_data.npy").T
    A = np.vander(x,n+1)
    
    # Way 1
    ATA = A.T@A
    x1 = la.inv(ATA)@A.T@b
    
    # Way 2
    Q,R = la.qr(A, mode='economic')
    x2 = la.solve_triangular(R,Q.T@b)
    
    #Plot it
    plt.scatter(x,b,s=12)
    plt.plot(x, np.polyval(x1,x), c='k',label = "Unstable")
    plt.plot(x, np.polyval(x2,x), c='g',label = "Stable")
    plt.legend(loc="upper right")
    plt.title("Least squares solutions of data")
    plt.show()
    
    #Return the forward errors
    return la.norm(A@x1-b,2), la.norm(A@x2-b,2)

#print(prob5(11))


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    #Initialize everything
    N = [i*5 for i in range(1,11)]
    x = sy.symbols('x')
    errors = []
    
    #Iterate through N
    for n in N:
        #Find the true value of the integral using sympy
        f = x**n*sy.exp(x-1)
        I = sy.integrate(f, (x,0,1))
        
        #Use the subfactorial method
        I2 = (-1)**n*(sy.subfactorial(n)-sy.factorial(n)/np.e)
        
        #Find and append the forward error
        error = np.abs(I-I2)
        errors.append(error)
        
    #Plot it
    plt.plot(N,errors)
    plt.yscale("log")
    plt.title("Forward error for subfactorial method")
    plt.xlabel("n-value")
    plt.show()
    
    return

#prob6()
