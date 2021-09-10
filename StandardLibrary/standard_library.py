# standard_library.py
"""Python Essentials: The Standard Library.
Bryant McArthur
Math 345
September 10
"""


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
    
    n = len(A)
    mylist = list()
    for i in range(n+1):
        mylist.append(set(it.combinations(A, i)))
    
    return mylist

print(power_set({'a','b','c'}))

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
    print(roll1)
    
    start = time.time()
    end = timelimit + time.time()
    
    while time.time() < end:
        print(mylist)
        if cl.sum(mylist) <= 6:
            roll1 = rdm.randint(1,6)
            print(roll1)
            
        else:
            roll1 = rdm.randint(1,6)
            roll2 = rdm.randint(1,6)
            print(cl.sum(roll1, roll2))
            
        break
    
    return


print(shut_the_box("Bryant",20))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    