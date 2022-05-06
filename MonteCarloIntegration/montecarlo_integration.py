# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
Bryant McArthur
Math 347
February 8, 2022
"""

import numpy as np
from scipy.stats import gamma, sem, mvn
from matplotlib import pyplot as plt

# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    
    X = np.random.uniform(0,1, size = (N,n))
    
    #Create an indicator variable list from those draws if norm is less than one
    ball = []
    for i in range(N):
        if np.sum(X[i]**2, axis = 0) <= 1:
            ball.append(1)
        else:
            ball.append(0)
    
    #Find the volume by multiplying by the number of "quadrants"
    proportion = np.sum(ball)/N
    
    volume = proportion * 2**n
    
    
    #Uses fancy equation I found for calculating Standard Error.
    SE = sem(ball)
                
    
    return volume

#print(ball_volume(2))


# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    
    #Create a linspace between a and b
    X = np.linspace(a,b,N)
    
    #approximate the integral
    V = b-a
    y = np.array([f(x) for x in X])
    
    summand = np.sum(y)
    
    approx = summand/N

    return approx*V


f = lambda x: x**2
#print(mc_integrate1d(f,-4,2,N=100))


# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    #Cast the mins and maxs as arrays and get X
    mins = np.array(mins)
    maxs = np.array(maxs)
    d = len(mins)
    X = np.random.uniform(0,1, size = (int(N),d))
    diffs = []
    
    # Find V
    diffs = maxs-mins
    V = np.prod(diffs)
    
    
    # Multiply the X and the differences elementwise and then add the mins
    X = np.multiply(X,diffs) + mins
    
    #Find the approximation
    y = [f(x) for x in X]
    summands = np.sum(y)
    approx = V/N*summands
    """
    
    points = np.random.uniform(0,1,(len(mins),N)).T*(diffs)+mins
    sum_vals = 0
    for i in range(N):
        sum_vals += f(points[i])
        
    approx = V*(sum_vals/N)
    """
    
    #print(approx)
    return approx


f = lambda x: x[0]**2+x[1]**2
f = lambda x: 3*x[0]-4*x[1]+x[1]**2
f = lambda x: x[0]+x[1]-x[3]*x[2]**2


# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    #Initialize n, the mins, and maxs
    n = 4
    mins = np.array([-3/2, 0, 0, 0])
    maxs = np.array([3/4, 1, 1/2, 1])
    
    #Initialize the means and covariance
    means, cov = np.zeros(n), np.eye(n)
    
    #Find the exact value of the integral
    exact = mvn.mvnun(mins,maxs,means,cov)[0]
    
    #Create a logspace
    X = np.logspace(1,5,20).astype(int)
    
    #Define f
    f = lambda x: (1. /(2*np.pi)**(x.size/2.)*np.exp(-(x@x)/2.))
    
    errors = []
    #Iterate through the Ns in X
    for N in X:
        #Get the estimate calling problem 3
        estimate = mc_integrate(f, mins, maxs, int(np.floor(N)))
        
        #Find the relative error and append
        error = np.abs(exact-estimate)/np.abs(exact)
        errors.append(error)
        
    #Define the comparison function and plot
    f = lambda N: 1/np.sqrt(N)
    plt.loglog(X, errors, label='Relative Error')
    plt.loglog(X, f(X), label= '$1/\sqrt{N}$')
    plt.legend(loc=0)
    plt.show()
    
    
#prob4()
    
