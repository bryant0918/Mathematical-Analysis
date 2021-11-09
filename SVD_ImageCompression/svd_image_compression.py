# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File."""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from imageio import imread

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    
    AH = A.conj().T
    eigvals, eigvecs = la.eig(AH@A)
    
    sigmas = []
    
    for i in range(len(eigvals)):
        if abs(eigvals[i]) > tol:
            sigmas.append(np.sqrt(eigvals[i]))
            
    sigmas = np.array(sigmas)
    
    idx = sigmas.argsort()[::-1]
    sigmas = sigmas[idx]
    eigvecs = eigvecs[:,idx]
    
    r = len(sigmas)
    
    V = eigvecs[:,:r]
    
    U = A@V / sigmas
    
    return U,sigmas,V.conj().T


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    
    theta = np.linspace(0,2*np.pi,200)
    for i in range(len(theta)):
        x = np.cos(theta)
        y = np.sin(theta)
    
    S = np.vstack((x,y))
    
    E = np.vstack((np.array([1,0]),np.array([0,0]),np.array([0,1])))
    E = E.T

    U, sigma, VH = la.svd(A)
    
    ax1 = plt.subplot(221)
    plt.plot(S[0],S[1])
    plt.plot(E[0],E[1])
    plt.title("S and E")
    plt.axis("equal")
    
    ax2 = plt.subplot(222)
    plt.plot((VH@S)[0],(VH@S)[1])
    plt.plot((VH@E)[0],(VH@E)[1])
    plt.title("VHS and VHE")
    plt.axis("equal")
    
    ax3 = plt.subplot(223)
    plt.plot((np.diag(sigma)@VH@S)[0],(np.diag(sigma)@VH@S)[1])
    plt.plot((np.diag(sigma)@VH@E)[0],(np.diag(sigma)@VH@E)[1])
    plt.title("SigmaVHS and SigmaVHE")
    plt.axis("equal")
    
    ax4 = plt.subplot(224)
    plt.plot((U@np.diag(sigma)@VH@S)[0],(U@np.diag(sigma)@VH@S)[1])
    plt.plot((U@np.diag(sigma)@VH@E)[0],(U@np.diag(sigma)@VH@E)[1])
    plt.title("USigmaVHS and USigmaVHE")
    plt.axis("equal")
    
    plt.tight_layout()
    plt.show()
    
    
    
    
# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    
    U,sig,VH = la.svd(A, full_matrices = False)
    
    if s > np.linalg.matrix_rank(A):
        raise ValueError("s is greater than the rank of A")
        
    sigs = sig[:s]
    Vs = VH[:s,:]
    Us = U[:,:s]
        
    As = Us @ np.diag(sigs) @ Vs
    
    entries = Us.size + sigs.size + Vs.size
    
    return As, entries


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    
    U, sig, VH = la.svd(A)
    
    if err <= sig[-1]:
        raise ValueError("Error is too small")
    
    i = 0
    while sig[i] >= err:
        i += 1
        
    As, entries = svd_approx(A,i-1)
    
    return As, entries


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    
    image = imread(filename) / 255
    gray = True
    
    ogsize = image.size
    
    if image.ndim == 3:
        gray = False
        
        redlayer = image[:,:,0]
        greenlayer = image[:,:,1]
        bluelayer = image[:,:,2]
        
        Ared, redsize = svd_approx(redlayer,s)
        Agreen, greensize = svd_approx(greenlayer, s)
        Ablue, bluesize = svd_approx(bluelayer, s)
        
        size = redsize + greensize + bluesize
        
        As = np.dstack((np.clip(Ared, 0,1),np.clip(Agreen,0,1),np.clip(Ablue,0,1)))
        
        
    else:
        As, size = svd_approx(image, s)
    
    if gray:
        plt.subplot(121)
        plt.imshow(image, cmap = "gray")
        plt.axis("off")
        plt.title("Original entries:" + str(ogsize))
        
        plt.subplot(122)
        plt.imshow(As, cmap = "gray")
        plt.axis("off")
        plt.title("Approximated entries:" + str(size))
        
        plt.suptitle("Approximating with " + str(s) + "Singular values")
        plt.show()
        
    else:
        plt.subplot(121)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Original entries:" + str(ogsize))
        
        plt.subplot(122)
        plt.imshow(As)
        plt.axis("off")
        plt.title("Approximated entries:" + str(size))
        
        plt.suptitle("Approximating with " + str(s) + " Singular values")
        plt.show()
        
    
    
    
if __name__ == "__main__":
    A = np.random.random((8,5))
    B = np.array([[2,0],[0,4]])
    u,sigma,v = compact_svd(A)
    #print(np.allclose(u.T @ u, np.identity(3)))
    #print(u,sigma,v)
    #print(np.allclose(u @ np.diag(sigma) @ v, B))
    
    #visualize_svd(A)
    #print(A)
    #print(lowest_rank_approx(A, .5))
    compress_image("hubble.jpg",20)
    
    
    pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
