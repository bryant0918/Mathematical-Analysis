# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
Bryant McArthur
Sec 002
My dad's birthday
"""
import sympy as sy
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    #Import symbols and create expression
    x,y = sy.symbols('x y')
    return sy.Rational(2,5)*sy.exp(x**2-y)*sy.cosh(x+y)+sy.Rational(3,7)*sy.log(x*y+1)

#print(prob1())

# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    #Import symbols, create expression and simplify
    x,j,i = sy.symbols('x j i')
    expr = sy.product(sy.summation(j*(sy.sin(x)+sy.cos(x)),(j,i,5)),(i,1,5))
    return sy.trigsimp(expr)

print(prob2())


# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    #Initialize variables
    x,n,y= sy.symbols('x,n,y')
    domain = np.linspace(-2,2,100)
    
    #Iterate through N to sum up
    for l in range(1,N+1):
        mac = sy.summation(x**n/sy.factorial(n),(n,0,l))
        expr = sy.lambdify(y,mac.subs(x,-y**2),"numpy")
        
    #Plot the Maclaurin Series and the actual function
    plt.plot(domain,expr(domain),label="Maclaurin Series")
    f = lambda y: np.exp(-y**2)
    plt.plot(domain,f(domain),label="Numpy")
    plt.title(f"Maclaurin Series up to degree {N} for exp(-y^2)")
    plt.ylim(0,1)
    plt.legend(loc='lower center')
    plt.show()
    

#prob3(10)
    
    


# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    #Initialize symbols and expression
    x,y,r,t = sy.symbols('x,y,r,t')
    rose = 1-((x**2+y**2)**sy.Rational(7,2)+18*x**5*y-60*x**3*y**3+18*x*y**5)/(x**2+y**2)**3
    
    #Substitute and symplify to get solution
    trig = rose.subs({x:r*sy.cos(t), y:r*sy.sin(t)})
    trigsimp = sy.trigsimp(trig)
    solution = sy.solve(sy.trigsimp(trigsimp))
    
    #Get r and lambdify for numpy
    f = solution[0].get(r)
    r_t = sy.lambdify(t,f,"numpy")
    
    domain = list(np.linspace(0,2*np.pi,200))
    
    #Plot it
    plt.plot(r_t(domain)*np.cos(domain),r_t(domain)*np.sin(domain) ,color = 'r')
    plt.title("My Rose")
    plt.show()
    
    
prob4()


# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    #Initialize variables and matrix
    x,y,l = sy.symbols('x,y,l')
    
    M = sy.Matrix([[x-y-l,x,0],
                   [x,x-y-l,x],
                   [0,x,x-y-l]])
    
    #Take the determinant and solve
    determinant = sy.det(M)
    solution = sy.solve(determinant,l)
    
    #Find the eigenvectors
    eigval1 = M.subs(l,solution[0])
    eigenvecs1 = eigval1.nullspace()[0]
    
    eigval2 = M.subs(l,solution[1])
    eigenvecs2 = eigval2.nullspace()[0]
    
    eigval3 = M.subs(l,solution[2])
    eigenvecs3 = eigval3.nullspace()[0]
    
    #Put them into a dictionary
    Eigen = {solution[0]:eigenvecs1,solution[1]:eigenvecs2,solution[2]:eigenvecs3}
    
    return Eigen

#print(prob5())


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    #Initialize variables and expression
    x = sy.symbols('x')
    p = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100
    p_x = sy.lambdify(x,p)
    mins = set()
    maxs = set()
    domain = np.linspace(-5,5,100)
    
    #Take the derivatives
    dp = sy.diff(p,x)
    ddf = sy.lambdify(x,sy.diff(dp,x))
    zeros = sy.solve(dp,x)
    
    #Iterate through zeros to see if min or max
    for z in zeros:
        if ddf(z) < 0:
            maxs.add(z)
        else:
            mins.add(z)
        
    #Plot it
    plt.plot(domain,[p_x(t) for t in domain])
    plt.title("p(x) with Extrema")
    
    for m in mins:
        plt.plot(m,p_x(m),'y',marker = 'o')
    for m in maxs:
        plt.plot(m,p_x(m),'k',marker = 'o')
        
    plt.show()
    
    return mins,maxs
        

#print(prob6())

# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    #Initialize everything
    x,y,z,p,t,phi,r = sy.symbols('x,y,z,p,theta,phi,r')
    f = (x**2+y**2+z**2)**2
    h1 = p*sy.sin(phi)*sy.cos(t)
    h2 = p*sy.sin(phi)*sy.sin(t)
    h3 = p*sy.cos(phi)
    h = sy.Matrix([[h1],[h2],[h3]])
    jacobian = h.jacobian([p,phi,t])
    I = f.subs({x:h1,y:h2,z:h3})
    
    #Integrate and symplify
    integral = sy.integrate(sy.simplify(I*sy.det(jacobian)),(p,0,r),(t,0,2*sy.pi),(phi,0,sy.pi))
    
    #Lambdify
    expr = sy.lambdify(r, integral)
    
    #Plot
    domain = np.linspace(0,3,1000)
    plt.plot(domain,[expr(d) for d in domain])
    plt.show()
    
    return expr(2)
    
#print(prob7())
    
    
