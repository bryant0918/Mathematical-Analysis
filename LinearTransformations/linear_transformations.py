# linear_transformations.py
"""Volume 1: Linear Transformations.
Bryant McArthur
Math 345 Sec 002
September 21
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import time


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    
    stretcher = np.array([[a,0],[0,b]])
    matrix = np.matmul(stretcher,A)
    
    return matrix
    

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    scale = np.array([[1,a],[b,1]])
    
    return np.matmul(scale,A)

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    c = 1/(a**2+b**2)
    scale = c*np.array([[(a**2)-(b**2),2*a*b],[2*a*b,(b**2)-(a**2)]])
    
    return np.matmul(scale,A)

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    
    scale = np.array([[np.cos(theta),-1*np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    
    return np.matmul(scale,A)


# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    
    Pe0 = np.array([x_e,0])
    Pm0 = np.array([x_m,0])
    
    start = time.perf_counter()
    final = start + T
    
    timer = np.linspace(0,T,500)
    
    
    PeT = np.array([rotate(Pe0,t*omega_e) for t in timer]).T
    PmT = np.array([rotate(Pm0 - Pe0,t*omega_m) for t in timer] + PeT.T).T
    
    plt.plot(PeT[0],PeT[1], 'b-', label="Earth")
    plt.plot(PmT[0],PmT[1], 'tab:orange',label="Moon")
    plt.legend(loc = "lower right")
    #plt.axis([-10,10,-10,10])
    plt.show()
    

#solar_system((3*np.pi)/2,10,11,1,13)    
    


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    
    n_times = [2**n for n in range(1,9)]
    mvtimes = [None] * 8
    mmtimes = [None] * 8
    print(n_times)
    
    for i in range(1,len(n_times)):
        print("i is ",i)
        v = random_vector(n_times[i])
        A = random_matrix(n_times[i])
        start = time.perf_counter()
        B = matrix_vector_product(A, v)
        mvtime = time.perf_counter()-start
        mvtimes[i] = mvtime
        B = random_matrix(n_times[i])
        start = time.perf_counter()
        C = matrix_matrix_product(A, B)
        mmtime = time.perf_counter() - start
        mmtimes[i] = mmtime
        
    ax1 = plt.subplot(121)
    ax1.plot(n_times,mvtimes, 'b.-', linewidth=1.5, markersize=10)
    plt.xlabel("n",fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    
    
    ax2 = plt.subplot(122)
    ax2.plot(n_times,mmtimes, '.-',color="orange", linewidth=1.5, markersize=10)
    plt.xlabel("n",fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    
    ax1.set_title("Matrix Vector", fontsize=8)
    ax2.set_title("Matrix Matrix", fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
#prob3()


# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    n_times = [2**n for n in range(1,9)]
    mvtimes = [None] * 8
    mmtimes = [None] * 8
    mmdots = [None] * 8
    mvdots = [None] * 8
    print(n_times)
    
    for i in range(1,len(n_times)):
        print("i is ",i)
        #Create Matrices and Vectors
        v = random_vector(n_times[i])
        A = random_matrix(n_times[i])
        
        #Matrix-Vector
        start = time.perf_counter()
        B = matrix_vector_product(A, v)
        mvtime = time.perf_counter()-start
        mvtimes[i] = mvtime
        
        #Matrix-Matrix
        B = random_matrix(n_times[i])
        start = time.perf_counter()
        C = matrix_matrix_product(A, B)
        mmtime = time.perf_counter() - start
        mmtimes[i] = mmtime
        
        #MV with @
        start = time.perf_counter()
        A = np.array(A) @ np.array(v)
        mvdot = time.perf_counter() - start
        mvdots[i] = mvdot
        
        #MM with @
        start = time.perf_counter()
        B = np.array(A) @ np.array(B)
        mmdot = time.perf_counter() - start
        mmdots[i] = mmdot
        
        
        
        
    ax1 = plt.subplot(121)
    ax1.plot(n_times,mvtimes, 'b.-', linewidth=1.5, markersize=10, label="Matrix-Vector")
    ax1.plot(n_times,mmtimes, '.-',color="orange", linewidth=1.5, markersize=10, label="Matrix-Matrix")
    ax1.plot(n_times,mvdots, 'k.-', linewidth=1.5, markersize=10, label="Matrix-Vector dot")
    ax1.plot(n_times,mmdots, '.-',color="yellow", linewidth=1.5, markersize=10, label="Matrix-Matrix dot")
    ax1.legend(loc="upper left")
    plt.xlabel("n",fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    
    
    ax2 = plt.subplot(122)
    ax2.loglog(n_times,mvtimes, 'b.-', linewidth=1.5, markersize=10, label="Matrix-Vector")
    ax2.loglog(n_times,mmtimes, '.-',color="orange", linewidth=1.5, markersize=10, label="Matrix-Vector")
    ax2.plot(n_times,mvdots, 'k.-', linewidth=1.5, markersize=10, label="Matrix-Vector dot")
    ax2.plot(n_times,mmdots, '.-',color="yellow", linewidth=1.5, markersize=10, label="Matrix-Matrix dot")
    ax2.legend(loc="upper left")
    plt.xlabel("n",fontsize=14)
    plt.ylabel("Seconds", fontsize=14)
    
    ax1.set_title("Linear", fontsize=8)
    ax2.set_title("Logrithimic", fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
prob4()

if __name__ == "__main__":
    """
    data = np.load("horse.npy")
    
    plt.plot(data[0],data[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    plt.show()
    
    #Subplot 1
    ax1 = plt.subplot(221)
    ax1.plot(stretch(data,.5,1.2)[0],stretch(data,.5,1.2)[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    #Subplot 2
    ax2 = plt.subplot(222)
    ax2.plot(shear(data,1.5,0)[0],shear(data,1.5,0)[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    #Subplot 3
    ax3 = plt.subplot(223)
    ax3.plot(reflect(data,0,1)[0],reflect(data,0,1)[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    #Subplot 4
    ax4 = plt.subplot(224)
    ax4.plot(rotate(data, np.pi/2)[0],rotate(data,np.pi/2)[1], 'k,')
    plt.axis([-1,1,-1,1])
    plt.gca().set_aspect("equal")
    
    #Set titles
    ax1.set_title("Stretched Horse", fontsize=8)
    ax2.set_title("Sheared Horse", fontsize=8)
    ax3.set_title("Reflected Horse", fontsize=8)
    ax4.set_title("Rotated Horse", fontsize=8)
    
    plt.suptitle("Horses")
    
    
    plt.show()
    """
    
    
    
    
    
    
    
    
    
    
    
    