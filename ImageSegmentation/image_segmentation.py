# image_segmentation.py
"""Volume 1: Image Segmentation.
Bryant McArthur
November 2
Sec 002
"""

import numpy as np
from scipy import linalg as la
from scipy import sparse as sp
from imageio import imread
from matplotlib import pyplot as plt

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    m,n = np.shape(A)
    rowsums = []
    
    for i in range(m):
        rowsums.append(sum(A[i,j] for j in range(m)))
        
    D = np.diag(rowsums)
    L = D - A
    
    return L
        


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    
    L = laplacian(A)
    
    eigs,v1 = la.eig(L)
    
    eigs = np.real(eigs)
    connected = 0
    
    for i in range(len(eigs)):
        if eigs[i] < tol:
            connected += 1

    np.sort(eigs)
    algcon = eigs[1]
    
    return connected, algcon


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        #Read and scale the matrix then store as an attribute
        image = imread(filename)
        scaled = image / 255
        self.im = scaled
        self.gray = False
        
        if image.ndim == 3:
            brightness = scaled.mean(axis=2)
            M,N = brightness.shape
            flat_brightness = np.ravel(brightness)
        
        else:
            self.gray = True
            flat_brightness = np.ravel(scaled)
            
        self.flatbright = flat_brightness
        

    # Problem 3
    def show_original(self):
        """Display the original image."""
        
        if self.gray == True:
            plt.imshow(self.im, cmap="gray")
            plt.axis("off")
        else:
            plt.imshow(self.im)
            plt.axis("off")
            
        

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        
        if self.gray == True:
            m,n = self.im.shape
            
        else:
            m,n,z = self.im.shape
        
        A = sp.lil_matrix(m*n,m*n)
        D = np.zeros([1,m*n])
        
        for index in range(m*n):
            indices, dist = get_neighbors(index, r, m, n)
            weights = []
            #print(self.flatbright(index))
            weights = [(0 - abs(self.flatbright[index] - self.flatbright(ind))/sigma_B2 - dist[i] / sigma_X2) for i, ind in enumerate(indices)]
            
            A[index, indices] = weights
            D[index] = sum(weights)
        
        return A, D

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        raise NotImplementedError("Problem 5 Incomplete")

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        raise NotImplementedError("Problem 6 Incomplete")


if __name__ == '__main__':
    A = np.array([[1,2],[5,0]])
    print(laplacian(A))
    print(connectivity(A))
    ImageSegmenter("dream_gray.png").show_original()
    ImageSegmenter("dream.png").adjacency()
#    ImageSegmenter("monument_gray.png").segment()
#    ImageSegmenter("monument.png").segment()
    pass
