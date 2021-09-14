# standard_library.py
"""Python Essentials: The Standard Library.
Bryant McArthur
Math 345
September 10
"""
import numpy as np
import box
import sys
import time
import random as rdm
import calculator as cl
import itertools as it

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
    ps = []
    
    for i in range(len(A)+1):
        newsubsets = [set(z) for z in it.combinations(A,i)]
        ps.extend(newsubsets)
    return ps

print(power_set([1,2,3]))


# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""
    
    #Define my variables
    mylist = [1,2,3,4,5,6,7,8,9]
    roll1 = 0
    roll2 = 0
    start = time.time()
    end = timelimit + time.time()
    mybool = False
    
    #Play the game as long as you have time left on the clock!
    while time.time() < end:
        """Roll the dice"""
        print("Numbers left: \t", mylist)
        if np.sum(mylist) <= 6:
            roll1 = rdm.randint(1,6)
            roll2 = 0
            print("Roll: \t", roll1)
            
        else:
            roll1 = rdm.randint(1,6)
            roll2 = rdm.randint(1,6)
            print("Roll: \t", cl.sum(roll1, roll2))
            
        roll = cl.sum(roll1, roll2)
        
        mybool = box.isvalid(roll,mylist) #check the validity of the dice roll.
        
        if mybool == False:
            print("Unlucky roll!")
            break
        
        timeleft = end - time.time()
        
        print("Seconds left: \t", (round(timeleft, ndigits = 2)))
        
        #Get the input from the user
        x = input("Numbers to eliminate: ")
        y = x.split( )
        for i in range(len(y)):
            y[i] = int(y[i])
        
        #Check the validity of the input
        while sum(y) != cl.sum(roll1, roll2):
            x = input("Invalid input, try again: ")
            y = x.split( )
            for i in range(len(y)):
                y[i] = int(y[i])
        
        #Remove the inputed numbers from the box.
        for i in range(len(y)):
            mylist.remove(y[i])
            
    timeplayed = round(time.time() - start, ndigits = 2)
    
    """Congratulate or mock the player accordingly"""
    if len(mylist) == 0:
        print("Score for player ", player, ": ", sum(mylist)," points")
        print("Time played: ", timeplayed, " seconds")
        print("Congratulations!! You shut the box!")
        
    else:
        print("Score for player ", player, ": ", sum(mylist)," points")
        print("Time played: ", timeplayed, " seconds")
        print("You're a loser!")
        
    return
    
"""Only run the program if the correct 3 arguments"""
if len(sys.argv) == 3:
    player = sys.argv[1]
    timelimit = int(sys.argv[2])
    shut_the_box(player,timelimit)
else:
    print("You didn't provide the correct arguments")

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    