# solutions.py
"""Volume 1: The Page Rank Algorithm.
Bryant McArthur
Sec 002
March 1
"""

import numpy as np
import re
import networkx as nx
import itertools

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        #Get size of A
        self.n = len(A)
        
        #If there are no labels the labels are integers
        if labels == None:
            self.labels = [i for i in range(self.n)]
        else:
            self.labels = labels
            
        #If the labels are not the write size then raise an error.
        if len(self.labels) != self.n:
            raise ValueError("wrong size")
        
        #If the colulmn is all zeros then set it to all ones
        for i in range(self.n):
            if np.all(A[:,i] == 0):
                A[:,i] = 1
        
        #Calculate and save Ahat
        Ahat = A / np.sum(A,axis=0)
        self.Ahat = Ahat
        
        
    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #Use equation to solve of form Ax=b
        P = np.linalg.solve(np.eye(self.n)-epsilon*self.Ahat,(1-epsilon)/self.n*np.ones(self.n))
        self.P = P
        
        #Create the dictionary
        dictionary = {self.labels[i]: p for i,p in enumerate(P)}
        
        return dictionary
        

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #Find B
        B = epsilon*self.Ahat + (1-epsilon)/self.n*np.ones((self.n,self.n))
        
        #Find eigenvalues and eigenvectors of B
        eigvals, eigvecs = np.linalg.eig(B)
        
        #Sort it this nifty way the TA gave
        P = eigvecs[:,np.argsort(eigvals.real)[-1]].real
        P = P/np.sum(P)
        
        #Create dictionary
        dictionary = {self.labels[i]: p for i,p in enumerate(P)}
        
        return dictionary
        

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # Calculate the first iteration
        P0 = np.ones(self.n).T/self.n
        P1 = epsilon*self.Ahat@P0 + (1-epsilon)/self.n*np.ones(self.n)
        t=0
        
        #Iterate while conditions meet
        while np.linalg.norm(P1-P0,ord=1) >= tol and t <= maxiter:
            P0 = P1.copy()
            P1 = epsilon*self.Ahat@P0 + (1-epsilon)/self.n*np.ones(self.n)
            t += 1
            
        #Normalize P and create dictionary
        P1 /= P1.sum()
        dictionary = {self.labels[i]: p for i,p in enumerate(P1)}
        
        return dictionary

# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    #Zip together the keys and values into a tuple
    l = [(-val,key) for key,val in zip(d.keys(), d.values())]
    #Sort lowest to highest of the negative values (essentially highest to lowest)
    l.sort()
    #Only return the keys
    return [i[1] for i in l]


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    #This will find the sequence of digits representing the websites
    pattern = re.compile(r"([\d]+)[\/|\n]",re.MULTILINE)
    
    #Open up the file and read it
    with open(filename, 'r') as myfile:
        data1 = myfile.read()
    
    #Find all the digits and create a set
    pages = pattern.findall(data1)  
    pages = set(pages)
    
    #Initialize empty matrix
    A = np.zeros((len(pages),len(pages)))
    
    #Create a list of pages and sort
    pagelist = list(pages)
    pagelist = list(np.sort(pagelist))
    
    #open up the file again to read by lines
    with open(filename, 'r') as myfile:
        data2 = myfile.readlines()
    
    #Find pattern on the line and link first every other page
    for line in data2:
        links = pattern.findall(line)
        i = pagelist.index(links[0])
        for link in links[1:]:
            A[pagelist.index(link),i] += 1
    
    #Use class to solve and get ranks
    di = DiGraph(A,pagelist)
    d = di.itersolve(epsilon)
    rank = get_ranks(d)
    
    return rank
    

# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """
    Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    #Define pattern to find the winner and loser
    pattern = re.compile(r"(.*),(.*)\n",re.MULTILINE)
    
    #Open the file and read
    with open(filename, 'r') as myfile:
        data = myfile.read()
    
    #Get all the games
    games = pattern.findall(data)
    
    #Initialize empty teams
    team1 = set()
    team2 = set()
    
    #Go through the games and add winners and losers to sets
    for game in games:
        team1.add(game[0])
        team2.add(game[1])
    
    #Get all the teams
    teams = set.union(team1, team2)
    
    #Initialize empty matrix
    A = np.zeros((len(teams),len(teams)))
    
    #Sort the list of teams
    teamlist = list(teams)
    teamlist = list(np.sort(teamlist))
    
    #Link the winner and loser
    for game in games:
        i = teamlist.index(game[0])
        A[i,teamlist.index(game[1])] += 1
    
    #Use class to solve and get rank
    di = DiGraph(A,teamlist)
    d = di.itersolve(epsilon)
    rank = get_ranks(d)
    
    return rank
    


# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    #Create DiGraph element and pattern
    DG = nx.DiGraph()
    pattern = re.compile(r"(.*)\n")
    
    """
    with open(filename, 'r', encoding = "utf-8") as myfile:
        for line in myfile:
            actors = line.strip().split('/')[1:]
            
            for actor1, actor2 in itertools.combinations(actors, 2):
                if DG.has_edge(actor2, actor1):
                    DG[actor2][actor1]["weight"] += 1
                else:
                    DG.add_edge(actor2, actor1, weight=1)
            
    """
    
    
    #Open and read the file
    with open(filename, 'r', encoding = "utf-8") as myfile:
        data = myfile.readlines()
    
    #Going by lines split the string by actors
    for line in data:
        line = line.split('/')
        line[-1] = pattern.sub(r"\1", line[-1])
        my_actors = line[1:]
        my_pairs = itertools.combinations(my_actors, 2)
    
        #Go through the actors in the pairs of combinations and add edges
        for actor1, actor2 in my_pairs:
                if DG.has_edge(actor2, actor1):
                    DG[actor2][actor1]["weight"] += 1
                else:
                    DG.add_edge(actor2, actor1, weight=1)
    
    
    
    return get_ranks(nx.pagerank(DG, alpha = epsilon))
    
    
    
if __name__ == "__main__":
    A = np.array([[0,0,0,0],
                  [1,0,1,0],
                  [1,0,0,1],
                  [1,0,1,0]])

    #dg = DiGraph(A, list(string.ascii_lowercase))
    #print(dg.linsolve())
    #print(dg.eigensolve())
    #print(dg.itersolve())
    #d = dg.itersolve()
    #print(get_ranks(d))
    #print(rank_websites(epsilon=.17)[:20])
    #print(rank_ncaa_teams("ncaa2010.csv")[:20])
    #print(rank_actors(epsilon = .7)[:3])
    #seventeen = ['98595', '32791', '178606', '64104', '96254', '177473', '230247', '203109', '217557', '68912', '28392', '77323', '92715', '26083', '130094', '99464', '12846', '106064', '332', '31328']
    #student17 = rank_websites(epsilon=.17)[:20]
    #print(seventeen == student17)
    #forty = ['98595', '32791', '178606', '64104', '96254', '28392', '77323', '92715', '26083', '130094', '99464', '12846', '106064', '332', '31328', '86049', '123900', '74923', '119538', '90571']
    #student40 = rank_websites(epsilon=.40)[:20]
    #print(forty == student40)
    #thirtysix = ['98595', '32791', '178606', '64104', '96254', '28392', '77323', '92715', '26083', '130094', '99464', '12846', '106064', '332', '31328', '86049', '123900', '74923', '119538', '90571']
    #student36 = rank_websites(epsilon=.36)[:20]
    #print(thirtysix == student36)
    #thirtyone = ['BYU', 'UConn', 'San Diego State', 'Coastal Carolina', 'Kansas', 'Kentucky', 'Butler', 'East Carolina', 'Kansas State', 'NJIT', 'Florida', 'Stephen F. Austin', 'Ohio State', "Saint Mary's (CA)", 'VCU', 'Texas', 'Duke', 'Morehead State', 'UNC', 'Utah State']
    #student31 = rank_ncaa_teams("ncaa2010.csv", epsilon=.31)[:20]
    #print(thirtyone == student31)
    #student = rank_ncaa_teams("ncaa2021.csv", epsilon=.75)[:64]
    #print(student)
    #actors = rank_actors(epsilon = .55)[:20]
    #correct = ['Leonardo DiCaprio', 'Robert De Niro', 'Tom Hanks', 'Al Pacino', 'Christian Bale', 'Jamie Foxx', 'Ben Kingsley', 'Christoph Waltz', 'Morgan Freeman', 'Tom Hardy', 'Ralph Fiennes', 'Matt Damon', 'Harrison Ford', 'Gary Oldman', 'Liam Neeson', 'Clint Eastwood', 'Brad Pitt', 'Kevin Spacey', 'Ryan Gosling', 'James Stewart']
    #print(actors == correct)
    #print(actors)
    #print(correct)
    
    """
    outcomes = []
    for i in [j*5 for j in range(1,20)]:
        print(i/100)
        
        student = rank_ncaa_teams("ncaa2021.csv", epsilon=i/100)[:100]
        print(student)
        outcomes.append(student)
        print()
        if i == 75:
            with open('bracket.txt', 'w') as output_file:
                for team in student:
                    output_file.write(team + '\n')
    
    print(outcomes[0] == outcomes[-1])
    """
    pass
    
    