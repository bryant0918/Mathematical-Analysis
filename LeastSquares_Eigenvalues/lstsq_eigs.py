# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
Bryant McArthur
Sec 002
October 26, 2021
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import cmath


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q,R = la.qr(A, mode="economic")
    x = la.solve_triangular(R, Q.T@b)
    
    return x

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    #Load contents
    contents = np.load("housing.npy")
    xk = contents[0:,0]
    ones = np.ones(len(xk))
    
    #Create matrices and vectors
    A = np.column_stack((xk, ones))
    b = contents[0:,1]
        
    x = least_squares(A,b)
    
    #Generate line of best fit
    y = x[0]*xk + x[1]
    
    #Plot the results
    plt.scatter(xk,b)
    plt.plot(xk,y, 'b-')
    plt.show()
    
        
    


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    #Load contents
    contents = np.load("housing.npy")
    xk = contents[0:,0]
    
    #Create polynomial A matrix
    A3 = np.vander(xk,4)
    A6 = np.vander(xk,7)
    A9 = np.vander(xk,10)
    A12 = np.vander(xk,13)
    
    #create vector b
    b = contents[0:,1]
    
    #Solve for vector x using least squares method
    x3 = la.lstsq(A3,b)[0]
    x6 = la.lstsq(A6,b)[0]
    x9 = la.lstsq(A9,b)[0]
    x12 = la.lstsq(A12,b)[0]
    
    #Fit the solultion to y
    y3 = [np.matmul(A3[i], x3) for i in range(len(A3))]
    y6 = [np.matmul(A6[i], x6) for i in range(len(A6))]
    y9 = [np.matmul(A9[i], x9) for i in range(len(A9))]
    y12 = [np.matmul(A12[i], x12) for i in range(len(A12))]
    
    """Plot the Data"""
    ax1 = plt.subplot(221)
    plt.scatter(xk,b, s = 5)
    ax1.plot(xk,y3,'b-', label = "Degree 3")
    plt.xlabel("years")
    plt.ylabel("price index")
    
    ax2 = plt.subplot(222)
    plt.scatter(xk,b, s = 5)
    ax2.plot(xk,y6, 'b-', label = "Degree 6")
    plt.xlabel("years")
    plt.ylabel("price index")
    
    ax3 = plt.subplot(223)
    plt.scatter(xk,b, s = 5)
    ax3.plot(xk,y9, 'b-', label = "Degree 9")
    plt.xlabel("years")
    plt.ylabel("price index")
    
    ax4 = plt.subplot(224)
    plt.scatter(xk,b, s = 5)
    ax4.plot(xk,y12, 'b-', label="Degree 12")
    plt.xlabel("years")
    plt.ylabel("price index")
    
    ax1.set_title("Degree 3", fontsize = 8)
    ax2.set_title("Degree 6", fontsize = 8)
    ax3.set_title("Degree 9", fontsize = 8)
    ax4.set_title("Degree 12", fontsize = 8)
    
    plt.suptitle("Lines of best fit")
    
    plt.tight_layout()
    plt.show()
    
#polynomial_fit()
    


def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")
    
#plot_ellipse(1,2,3,4,5)

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    
    #Load the data
    xk, yk = np.load("ellipse.npy").T
    #Create function of least squares matrices
    A = np.column_stack((xk**2, xk, xk*yk, yk, yk**2))
    b = np.ones(len(xk))
    
    a,b,c,d,e = la.lstsq(A,b)[0]
    
    #Plot the results using plot_ellipse function
    plt.scatter(xk,yk, s=5, color='k')
    plot_ellipse(a,b,c,d,e)
    plt.suptitle("Data fitted to ellipse")
    plt.show()


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    
    m,n = A.shape
    x = np.random.rand(n)
    #Normalize x
    x = x/la.norm(x)
    for k in range(N):
        y = x
        x = A @ x
        x = x/la.norm(x)
        if la.norm(x - y) < tol:
            break
    
    return x.T @ A @ x, x


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    
    m,n = A.shape
    S = la.hessenberg(A)
    #Recombine the QR decomposition of A_k into A_k+1
    for k in range(N):
        Q,R = la.qr(S)
        S = R@Q
    #Initialize values
    eigs = []
    i = 0
    #Iterate over the columns of A
    while i < n:
        #If S_i is 1x1
        if S[i][i] == S[n-1][n-1] or abs(S[i+1][i]) < tol:
            eigs.append(S[i][i])
        #Calculate the eigenvalues of S_i and append
        else:
            b = - (S[1][1] + S[2][2])
            c = la.det([[S[1][1],S[1][2]],[S[2][1],S[2][2]]])
            
            e1 = (-b + cmath.sqrt(b**2 - 4*c))/(2)
            e2 = (-b - cmath.sqrt(b**2 - 4*c))/(2)
            
            eigs.append(e1)
            eigs.append(e2)
            i += 1
        i += 1
            
    return eigs

