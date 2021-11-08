# image_segmentation.py
"""Volume 1: Image Segmentation.
Bryant McArthur
November 2
Sec 002
"""

import numpy as np
from scipy import linalg as la
from imageio import imread
from matplotlib import pyplot as plt
from scipy.sparse import csc_matrix, lil_matrix, diags, csgraph, linalg

# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    #Don't let this fool you, A is a square matrix
    m,n = np.shape(A)
    rowsums = []
    
    for i in range(m):
        rowsums.append(sum(A[i,j] for j in range(m)))
        
    #Put the row sums on the diagonal
    D = np.diag(rowsums)
    #Find the Laplacian
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
    #Get the Laplacian of A
    L = laplacian(A)
    
    #Get the eigenvalues and irrelevant eigenvectors
    eigs,v1 = la.eig(L)
    
    #Make the eigenvalues real
    eigs = np.real(eigs)
    connected = 0
    
    #Iterate through the eigenvalues to see if they're close enough to zero
    for i in range(len(eigs)):
        if eigs[i] < tol:
            connected += 1
    
    #Sort the eigenvalues and take the second argument for the algebraic connectivity
    eigs = np.sort(eigs)
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
        #Store whether it's gray or not as an attribute
        self.gray = False
        
        #If Color scale and ravel
        if image.ndim == 3:
            brightness = scaled.mean(axis=2)
            M,N = brightness.shape
            flat_brightness = np.ravel(brightness)
        
        #If gray set attribute to gray and ravel
        else:
            self.gray = True
            flat_brightness = np.ravel(scaled)
            
        #Set raveled as attribute
        self.flatbright = flat_brightness
        

    # Problem 3
    def show_original(self):
        """Display the original image."""
        
        #If gray show with cmap = "gray"
        if self.gray == True:
            plt.imshow(self.im, cmap="gray")
            plt.axis("off")
        
        #If color show normal
        else:
            plt.imshow(self.im)
            plt.axis("off")
            
        

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        
        #Get the shape whether it's gray or color
        if self.gray == True:
            m,n = self.im.shape
        else:
            m,n,z = self.im.shape
        
        #Initialize A and D
        A = lil_matrix((m*n,m*n))
        D = np.zeros(m*n)
        
        
        for index in range(m*n):
            indices, dist = get_neighbors(index, r, m, n)
            weights = []
            
            #Iterate through both the indice neighbors and the distances at the same time
            for ind, k in zip(indices, dist):
                
                #Use the handy weight equation they gave us
                if k < r:
                    weight = np.exp(0-abs(self.flatbright[index] - self.flatbright[ind])/sigma_B2 - k / sigma_X2)
            
                else:
                    weight = 0
                    
                weights.append(weight)
                
            D[index] = sum(weights)
            
            A[index, indices] = weights
            
        #Return a csc matrix
        A = csc_matrix(A)
        
        return A, D

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        
        #Get the dimensions for when you reshape it
        if self.gray == True:
            m,n = self.im.shape
        else:
            m,n,z = self.im.shape
        
        #Find the laplacian and the weird D^(-.5) matrix
        L =csgraph.laplacian(A)
        Dneg =diags(D**(-.5))
        
        #Calculate value
        x = Dneg @ L @ Dneg
        
        #Get the eigen values and eigenvectors
        eigvals, eigvecs = linalg.eigsh(x, which = "SM", k = 2)
        eigvec = eigvecs[:,1]
        
        #Define the mask
        mask = np.reshape(eigvec, (m,n)) > 0
        
        return mask
        
        

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        
        #call Adjacency
        A, D = self.adjacency(r, sigma_B, sigma_X)
        
        #call cut
        mask = self.cut(A,D)
        
        #If gray
        if self.gray == True:
            #Define positive and negative images
            positive = np.multiply(self.im, mask)
            negative = np.multiply(self.im, ~mask)
            
            #Original image
            ax1 = plt.subplot(131)
            ax1.imshow(self.im, cmap = "gray")
            plt.title("Original")
            ax1.axis("off")
            
            #Positive image
            ax2 = plt.subplot(132)
            ax2.imshow(positive, cmap="gray")
            plt.title("Positive")
            ax2.axis("off")
            
            #Negative image
            ax3 = plt.subplot(133)
            ax3.imshow(negative, cmap="gray")
            plt.title("Negative")
            ax3.axis("off")
            
        else:
            #Stack them mask so that it works for color
            mask = np.dstack((mask,mask,mask))
            #Define positive and negative images
            positive = np.multiply(self.im, mask)
            negative = np.multiply(self.im, ~mask)
            
            #Original image
            ax1 = plt.subplot(131)
            ax1.imshow(self.im)
            plt.title("Original")
            ax1.axis("off")
            
            #Positive image
            ax2 = plt.subplot(132)
            ax2.imshow(positive)
            plt.title("Positive")
            ax2.axis("off")
            
            #Negative image
            ax3 = plt.subplot(133)
            ax3.imshow(negative)
            plt.title("Negative")
            ax3.axis("off")
        
        #plt.show()
        


if __name__ == '__main__':
    A = np.array([[1,2],[5,0]])
    #print(laplacian(A))
    #print(connectivity(A))
    A, D = ImageSegmenter("dream_gray.png").adjacency()
    #print(ImageSegmenter("dream_gray.png").cut(A,D))
    #print("A,D", A,D)
    print(ImageSegmenter("dream.png").segment())
    a = np.load("HeartMatrixA.npz")
    d = np.load("HeartMatrixD.npy")
    #print(a,A)
    #print(np.allclose(a,A))
    #print(d,D)
    #print(np.allclose(A, a))
    #print(np.allclose(D, d))
    #print(ImageSegmenter("monument_gray.png").cut())
#    ImageSegmenter("monument.png").segment()
    pass
