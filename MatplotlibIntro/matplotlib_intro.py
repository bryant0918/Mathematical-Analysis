# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
Bryant McArthur
Math 345
Sep 14, 2021
"""

import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    A = np.random.normal(size=(n,n))
    means = np.mean(A, axis=1)
    
    return np.var(means)
    

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    n = []
    for i in range(11):
        n = np.append(n,var_of_means(i*100))
        
    plt.plot(n)
    plt.show()
    

# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    #Define x and y
    x = np.linspace(-2*np.pi,2*np.pi,100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.arctan(x)
    
    #plot functions
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    
    plt.show()
    


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    #Define x and y
    x1 = np.linspace(-2,1,100)
    x2 = np.linspace(1,6,100)
    y1 = 1/(x1-1)
    y2 = 1/(x2-1)
    
    #plot functions
    plt.plot(x1,y1, '--m', linewidth = 4)
    plt.plot(x2,y2, '--m', linewidth = 4)
    plt.xlim(-2,6)
    plt.ylim(-6,6)
    
    plt.show()
    

# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    #Define x and y
    x = np.linspace(0,2*np.pi)
    y1 = np.sin(x)
    y2 = np.sin(2*x)
    y3 = 2*np.sin(x)
    y4 = 2*np.sin(2*x)
    
    #Subplot 1
    ax1 = plt.subplot(221)
    ax1.plot(x,y1, '-g')
    plt.xlim(0,2*np.pi)
    plt.ylim(-2,2)
    #Subplot 2
    ax2 = plt.subplot(222)
    ax2.plot(x,y2, '--r')
    plt.xlim(0,2*np.pi)
    plt.ylim(-2,2)
    #Subplot 3
    ax3 = plt.subplot(223)
    ax3.plot(x,y3, '--b')
    plt.xlim(0,2*np.pi)
    plt.ylim(-2,2)
    #Subplot 4
    ax4 = plt.subplot(224)
    ax4.plot(x,y4, 'm:')
    plt.xlim(0,2*np.pi)
    plt.ylim(-2,2)
    
    #Set titles
    ax1.set_title("sin(x)", fontsize=8)
    ax2.set_title("sin(2x)", fontsize=8)
    ax3.set_title("2sin(x)", fontsize=8)
    ax4.set_title("2sin(2x)", fontsize=8)
    
    plt.suptitle("Graphs of sin(x)")
    
    
    plt.show()
    


# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    data = np.load('FARS.npy')
    
    #Subplot 1
    ax1 = plt.subplot(121)
    ax1.plot(data[:,1],data[:,2], 'k.', markersize=.01)
    plt.axis("equal")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    
    #Subplot 2
    ax2 = plt.subplot(122)
    ax2.hist(data[:,0], bins = 24, range=[0,25], ec='black')
    plt.xlabel("Hour of the Day")
    plt.xlim(1,24)
    
    plt.suptitle("Car Crashes in the USA")
    
    plt.show()
    


# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    #Define x, y and f(x,y)
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = np.linspace(-2*np.pi, 2*np.pi, 100)
    X, Y = np.meshgrid(x, y)
    fx = (np.sin(X)*np.sin(Y))/(X*Y)
    
    #Heat Map
    plt.subplot(121)
    plt.pcolormesh(X,Y,fx,cmap="magma")
    plt.colorbar()
    
    #Contour Map
    plt.subplot(122)
    plt.contour(X,Y,fx,10, cmap="coolwarm")
    plt.colorbar()
    
    plt.suptitle("Heat Map and Contour Map of f(x,y)=(sin(x)sin(y))/xy")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
