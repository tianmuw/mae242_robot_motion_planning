#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:12:40 2020

@author: sonia, parth
"""



from gridWorld import *

from nextState import nextState
from smallGrid import smallGrid 
from mediumGrid import mediumGrid
from testGrid import testGrid
from costFunction import getCost, getCostBridge
import numpy as np
import copy


"""
Implement your value iteration algorithm
"""
# uSet = [(1,0),(0,1),(-1,0),(0,-1)]

def valueIteration(gamma,cost,eta,gridname):
    """
    Implement value iteration with a discount factor gamma and 
    the pre-defined cost functions in this assignment. It is passed as a
    string argument above. 
    Output:
    values: Numpy array of (n,m) dimensions
    policy: Numpy array of (n,m) dimensions
    """
    error = 1e-3
    values = None 
    policy = None
    iterations = None
    
    return values, policy, iterations

"""
Implement your policy iteration algorithm
"""

def policyIteration(gamma,cost,eta,gridname):
    """
    (Offline) Policy iteration with a discount factor gamma and 
    pre-defined cost functions. 
    Output:
    values: Numpy array of (n,m) dimensions
    policy: Numpy array of (n,m) dimensions
    """
    error = 1e-3
    values = None 
    policy = None
    iterations = None
    
    return values, policy, iterations

def optimalValues(question):
    """
    Please input your values of gamma and eta
    for each assignment problem here.
    """
    if question=='a':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='b':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='c':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='d1':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='d2':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='d3':
        gamma=0.9
        eta=0.2
        return gamma, eta
    elif question=='d4':
        gamma=0.9
        eta=0.2
        return gamma, eta
    else: 
        pass
    return 0
    
def showPath(xI,xG,path,n,m,O):
    gridpath = makePath(xI,xG,path,n,m,O)
    fig, ax = plt.subplots(1, 1) # make a figure + axes
    ax.imshow(gridpath) # Plot it
    ax.invert_yaxis() # Needed so that bottom left is (0,0)
    
    
# Function to actually plot the cost-to-gos
def plotValues(values,xI,xG,n,m,O):
    gridvalues = makeValues(values,xI,xG,n,m,O)
    fig, ax = plt.subplots() # make a figure + axes
    ax.imshow(gridvalues) # Plot it
    ax.invert_yaxis() # Needed so that bottom left is (0,0)


def showValues(n,m,values,O):
    string = '------'
    for i in range(0, n):
        string = string + '-----'
    for j in range(0, m):
        print(string)
        out = '| '
        for i in range(0, n):            
            jind = m-j-1 # Need to reverse index so bottom-left is (0,0)
            if isObstacle((i,jind),O):
                out += 'Obs' + ' | '
            else:
                out += str(values[i,jind]) + ' | '
        print(out)
    print(string)    


if __name__ == '__main__':
    gridname='small'
    # Use small and medium grid for your code submission
    # cost types: {'cost', 'bridge'}
    if gridname=='small':
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    elif gridname=='medium':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
    # elif gridname=='test':
    #     n, m, O, START, WINSTATE, DISTANTEXIT, LOSESTATE, LOSESTATES = testGrid()
    else:
        raise NameError("Unknown grid")
    
    # values, policy = valueIteration()
    # values, policy = policyIteration()
    """
    # Case 1:
    """
    gridname = 'medium'
    # values, policy = policyIteration() 
    """
    #Case 2
    """
    """
    # Case 3
    """
    """
    # Case 4
    """
    
    # Sample use of plotValues from gridWorld
    values = np.zeros((n,m))
    # Loop through values to just assign some dummy/arbitrary data
    for i in range(n):
        for j in range(m):
            if not(isObstacle((i,j),O)):
                values[i][j] = (n+2*m-2) - (i + 2*j)
    # This will print those numeric values as console text
    showValues(n,m,values,O)

    # This will plot the actual grid with objects as black and values as
    # shades from green to red in increasing numerical order
    xI = [1,1]
    xG = [4,3]
    grid = create_binary_grid(n, m, O)
    plotValues(grid*values,xI,xG,n,m,O)
    path = [[2,1],[3,1],[4,1],[4,2],[4,3]]
    showPath(xI,xG,path,n,m,O)

    

    # plt.show()
