# standard_library.py
"""Python Essentials: The Standard Library.
Bryant McArthur
Math 345
September 10
"""
import numpy as np

# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return min(L),max(L),(sum(L)/len(L))


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    mylist = [2,3,4,5]
    myint = 4
    mystr = "Hello world!"
    mytpl = (2,3,4,5)
    myset = {2,3,4,5}

    newint = myint
    newstr = mystr
    newlist = mylist
    newtpl = mytpl
    newset = myset
    
    newint += 1
    newstr += "Goodbye world!"
    newlist.append('b')
    newtpl += (1,)
    newset.add(2)
    
    if myint == newint:
        print("int is mutable")
    else:
        print("int is immutable")
    if newstr == mystr:
        print("strings are mutable")
    else:
        print("strings are immutable")
    if newlist == mylist:
        print("lists are mutable")
    else:
        print("lists are immutable")   
    if newtpl == mytpl:
        print("tuples are mutable")
    else:
        print("tuples are immutable")
    if newset == myset:
        print("sets are mutable")
    else:
        print("sets are immutable")
        
    return

print(prob2())

# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt that are 
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    import calculator as cl
    
    return (cl.sqrt(cl.sum(cl.product(a,a),cl.product(b,b))))
    
print(hypot(2,3))


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    import itertools as it
    """
    n = len(A)
    mylist = []
    for i in range(n+1):
        mylist.append(list(it.combinations(A, i)))
    
    return mylist
    """
    ps = [set()]
    
    for i in A:
        newsubsets = [set(z) for z in it.combinations(A,i)]
        ps.extend(newsubsets)
    return ps

print(power_set({1,2,3}))



# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    import time
    import random as rdm
    import calculator as cl
    
    mylist = [1,2,3,4,5,6,7,8,9]
    die = [1,2,3,4,5,6]
    roll1 = 0
    roll2 = 0
    start = time.time()
    end = timelimit + time.time()
    
    while time.time() < end:
        print("Numbers left: \t", mylist)
        if np.sum(mylist) <= 6:
            roll1 = rdm.randint(1,6)
            roll2 = 0
            print("Roll: \t", roll1)
            
        else:
            roll1 = rdm.randint(1,6)
            roll2 = rdm.randint(1,6)
            print("Roll: \t", cl.sum(roll1, roll2))
            
        print(roll1, roll2)
        roll = cl.sum(roll1, roll2)
        
        pwrset = power_set(mylist)
        mybool = False
        
        for i in range(len(mylist)):
            if mylist[i] == roll:
                mybool = True
        
        for i in range(len(pwrset)):
            if sum(pwrset[i]) == roll:
                mybool = True
                break
        
        if mybool == False:
            print("Unlucky roll!")
            break
        
            
        timeleft = end - time.time()
        
        print("Seconds left: \t", (round(timeleft, ndigits = 2)))
        
        x = input("Numbers to eliminate: ")
        y = x.split( )
        for i in range(len(y)):
            y[i] = int(y[i])
        
        while sum(y) != cl.sum(roll1, roll2):
            x = input("Invalid input, try again: ")
            y = x.split( )
            for i in range(len(y)):
                y[i] = int(y[i])
        
        
        for i in range(len(y)):
            mylist.remove(y[i])
            
    timeplayed = round(time.time() - start, ndigits = 2)
    
    if len(mylist) == 0:
        print("Score for player ", player, ": ", sum(mylist)," points")
        print("Time played: ", timeplayed, " seconds")
        print("Congratulations!! You shut the box!")
        
    else:
        print("Score for player ", player, ": ", sum(mylist)," points")
        print("Time played: ", timeplayed, " seconds")
        print("You're a loser!")
        
    
    return


print(shut_the_box("Bryant",60))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    