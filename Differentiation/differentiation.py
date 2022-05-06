# differentiation.py
"""Volume 1: Differentiation.
Bryant McArthur
Sec 002
January 18, 2022
"""

import numpy as np
from matplotlib import pyplot as plt
import sympy as sy
from autograd import numpy as anp
from autograd import grad,elementwise_grad
import time

# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    x = sy.symbols('x')
    expr = (sy.sin(x)+1)**(sy.sin(sy.cos(x)))
    expr = sy.diff(expr,x)
    expr = sy.lambdify(x, expr,'numpy')
    
    """
    domain = np.linspace(0-np.pi,np.pi,200)
    plt.plot(domain,expr(domain))
    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    """
    
    return expr
    
    
#print(prob1())


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x + h) - f(x))/h


"""
domain = np.linspace(0-np.pi,np.pi,200)
plt.plot(domain,fdq1(prob1(),domain))
ax = plt.gca()
ax.spines["bottom"].set_position("zero")
plt.ylim(-8,2)
plt.title("fdq1")
plt.show()
"""

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-3*f(x)+4*f(x+h)-f(x+2*h))/(2*h)

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x)-f(x-h))/h

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3*f(x)-4*f(x-h)+f(x-2*h))/(2*h)

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return (f(x+h)-f(x-h))/(2*h)

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (f(x-(2*h))-(8*f(x-h))+8*f(x+h)-f(x+(2*h)))/(12*h)


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    #Initialize everything
    domain = np.logspace(-8,0,9)
    x = sy.symbols('x')
    f = (sy.sin(x)+1)**(sy.sin(sy.cos(x)))
    f = sy.lambdify(x,f,'numpy')
    fx = prob1()
    fx = fx(x0)
    
    #Initialize lists
    fdq1x = []
    fdq2x = []
    bdq1x = []
    bdq2x = []
    cdq2x = []
    cdq4x = []
    
    #For different values of h append the error
    for h in domain:
        fdq1x.append(np.abs(fx-fdq1(f,x0,h)))
        fdq2x.append(np.abs(fx-fdq2(f,x0,h)))
        bdq1x.append(np.abs(fx-bdq1(f,x0,h)))
        bdq2x.append(np.abs(fx-bdq2(f,x0,h)))
        cdq2x.append(np.abs(fx-cdq2(f,x0,h)))
        cdq4x.append(np.abs(fx-cdq4(f,x0,h)))
        
    #Plot it
    plt.loglog(domain,fdq1x,'-o', label = "Order 1 Forward")
    plt.loglog(domain,fdq2x,'-o', label = "Order 2 Forward")
    plt.loglog(domain,bdq1x,'-o', label = "Order 1 Backward")
    plt.loglog(domain,bdq2x,'-o', label = "Order 2 Backward")
    plt.loglog(domain,cdq2x,'-o', label = "Order 2 Centered")
    plt.loglog(domain,cdq4x,'-o', label = "Order 4 Centered")
    plt.legend(loc = 'upper left',prop = {'size' : 9})
    plt.ylabel("Absolute Error")
    plt.xlabel("h")
    plt.show()
    
    
#prob3(1)
    


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    #Load data
    data = np.load("plane.npy")
    
    #Convert to radians
    data[:,1] = np.deg2rad(data[:,1])
    data[:,2] = np.deg2rad(data[:,2])
    
    #Initialize lists
    x,y = [],[]
    xprime,yprime = [],[]
    speed = []
    
    #Iterate through to make x and y coordinates with formula
    for i in range(0,8):
        x.append(500 * np.tan(data[i,2])/(np.tan(data[i,2])-np.tan(data[i,1])))
        y.append(500 * np.tan(data[i,2])*np.tan(data[i,1])/(np.tan(data[i,2])-np.tan(data[i,1])))
        
    #Find x prime from three different ways
    xprime = [x[1]-x[0]]
    yprime = [y[1]-y[0]]
    for i in range(1,7):
        xprime.append(.5*(x[i+1]-x[i-1]))
        yprime.append(.5*(y[i+1]-y[i-1]))
    xprime.append(x[7]-x[6])
    yprime.append(y[7]-y[6])
    
    #append these to speed
    for i in range(8):
        speed.append(np.sqrt((xprime[i]**2)+(yprime[i]**2)))
    
    return speed
    
    
#print(prob4())


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    #Initialize everything
    I = np.identity(len(x))
    dfdx = []
    
    #Go through x and find dfdx columns
    for j in range(len(x)):
        dfdx.append((f(x+h*I[j])-f(x-h*I[j]))/(2*h))
        
    #They were actually rows so make them arrays and transpose
    dfdx = np.array(dfdx)
    Jacobian = dfdx.T
    
    return Jacobian

f = lambda x: np.array([x[0]**2, x[0]**3 -x[1]])
#print(jacobian_cdq2(f,[1,2]))


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    #Make recursive formula
    if n ==0:
        return anp.ones_like(x)
    elif n==1:
        return x
    else:
        return 2*x*cheb_poly(x,n-1)-cheb_poly(x,n-2)
       
        
     

def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    #Initialize the grad
    d_cheb = elementwise_grad(cheb_poly)
    domain = np.linspace(-1,1,100)
    
    #Iterate through and plot on subplots
    for n in range(0,5):
        k = n+1
        plt.subplot(2,3,k)
        plt.plot(domain,d_cheb(domain,n))
        plt.title(f"n={n}")
        plt.ylim(-5,5)
        
    #Finish plotting
    plt.suptitle("Derivatives of Chebyshev Polynomials")
    plt.tight_layout()
    plt.show()
    
    
    return

#prob6()


# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    #Initialize everything
    sympy = []
    diff, differr = [],[]
    auto,autoerr = [],[]
    g = lambda x: (anp.sin(x)+1)**anp.sin(anp.cos(x))
    x = sy.symbols('x')
    expr = (sy.sin(x)+1)**(sy.sin(sy.cos(x)))
    expr = sy.lambdify(x, expr,'numpy')
    
    #Iterate through N
    for i in range(N):
        x0 = np.random.rand()
        
        #Compute time for Sympy
        start = time.perf_counter()
        f = prob1()
        fx = f(x0)
        sympy.append(time.perf_counter()-start)
        
        #Compute Difference Quotient time and error
        start = time.perf_counter()
        f1 = cdq4(expr,x0)
        diff.append(time.perf_counter()-start)
        differr.append(np.abs(fx - f1))
        
        #Compute Autograd time and error
        start = time.perf_counter()
        dg = grad(g)
        dgx = dg(x0)
        auto.append(time.perf_counter()-start)
        autoerr.append(np.abs(fx - dgx))
        
    #Plot it
    plt.scatter(sympy,anp.ones_like(sympy)*1e-18,alpha = 0.35,label="Sympy")
    plt.scatter(diff,differr,alpha = .35,label="Difference Quotients")
    plt.scatter(auto,autoerr,alpha = .35,label= "Autograd")
    plt.xlabel("Computation Time")
    plt.ylabel("Absolute Error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc=0)
    plt.suptitle("Differentiation Methods")
    plt.show()
        
    return


prob7()