#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Tianmu Wang (tiw028@ucsd.edu)
"""

# You are free to use these problems for educational purposes.


import numpy as np
import matplotlib.pyplot as plt
from code.mazemods import maze
from code.mazemods import canvisit
from code.mazemods import collisionCheck
from code.mazemods import makePath
from code.mazemods import getPathFromActions
from code.mazemods import getCostOfActions
from code.mazemods import stayWestCost
from code.mazemods import stayEastCost


def depthFirstSearch(xI, xG, n, m, O):
    """
      Search the deepest nodes in the search tree first.
    """
    controls = ((1, 0), (-1, 0), (0, 1), (0, -1))
    stack = [xI]  # Create a stack
    visited = set()
    parents = {xI: None}
    actions = list()

    while len(stack) > 0:
        node = stack.pop()  # Get the node
        visited.add(node)  # Mark it as visited
        if node == xG:  # Find the terminal node

            s = xG
            while parents[s] is not None:
                action = tuple(map(lambda i, j: i - j, s, parents[s]))  # Compute the action
                s = parents[s]  # Search its parent
                actions.insert(0, action)  # Get the action list
            total_cost = getCostOfActions(xI, actions, O)  # Compute the total cost
            num_visited = len(visited)  # Get the number of visited nodes
            return actions, total_cost, num_visited
        else:
            for control in controls:  # For all inputs
                collided = collisionCheck(node, control, O)  # Check whether it is collided
                if not collided:  # The input is valid
                    child = tuple(map(lambda i, j: i + j, node, control))  # Get the next node
                    if child not in visited:
                        stack.append(child)
                        parents[child] = node
    return False


def breadthFirstSearch(xI, xG, n, m, O):
    """
    Search the shallowest nodes in the search tree first [p 85].
    """

    controls = ((1, 0), (-1, 0), (0, 1), (0, -1))
    queue = [xI]  # Create a queue
    visited = set()
    parents = {xI: None}
    actions = list()

    while len(queue) > 0:
        node = queue.pop(0)
        visited.add(node)
        if node == xG:  # Find the goal node
            s = xG
            while parents[s] is not None:
                action = tuple(map(lambda i, j: i - j, s, parents[s]))  # Compute the action
                s = parents[s]
                actions.insert(0, action)
            total_cost = getCostOfActions(xI, actions, O)  # Obtain the total cost
            num_visited = len(visited)  # Obtain the number of visited nodes
            return actions, total_cost, num_visited
        else:
            for control in controls:  # For all inputs
                collided = collisionCheck(node, control, O)
                if not collided:  # The input is valid
                    child = tuple(map(lambda i, j: i + j, node, control))
                    if child not in visited:
                        queue.append(child)
                        parents[child] = node
    return False


def DijkstraSearch(xI, xG, n, m, O, cost='westCost'):
    """
    Search the nodes with least cost first.
    """
    if cost == 'westcost':
        cost_func = stayWestCost
    elif cost == 'eastcost':
        cost_func = stayEastCost
    else:
        return False

    controls = ((1, 0), (-1, 0), (0, 1), (0, -1))
    pqueue = list()  # Create a priority queue
    actions = list()
    cost = cost_func(xI, actions, O)
    best_cost_library = {xI: cost}
    pqueue.append((cost, xI))
    visited = set(xI)
    parents = {xI: None}

    while len(pqueue) > 0:
        pqueue.sort(key=lambda pair: pair[0])  # Sort the list
        pair = pqueue.pop(0)  # Get the node with the smallest cost
        node = pair[1]
        if node == xG:
            s = xG
            while parents[s] is not None:
                action = tuple(map(lambda i, j: i - j, s, parents[s]))
                s = parents[s]
                actions.insert(0, action)
            total_cost = cost_func(xI, actions, O)  # Compute the total cost
            num_visited = len(visited)  # Obtain the number of visited nodes
            return actions, total_cost, num_visited
        else:
            for control in controls:
                collided = collisionCheck(node, control, O)
                if not collided:
                    child = tuple(map(lambda i, j: i + j, node, control))
                    if child != parents[node]:  # Avoid the repeated path
                        cost = cost_func(node, [control], O) + best_cost_library[node]
                        if child not in visited:
                            best_cost_library[child] = cost  # Update its optimal cost
                            parents[child] = node  # Update its parent
                            visited.add(child)  # Mark it as visited
                            pqueue.append((best_cost_library[child], child))
                        else:
                            if cost < best_cost_library[child]:
                                best_cost_library[child] = cost  # Update its optimal cost
                                parents[child] = node  # Update its parent
                                pqueue.append((best_cost_library[child], child))
                    else:
                        continue
    return False


def nullHeuristic(state, goal):
    """
      A heuristic function estimates the cost from the current state to the nearest
      goal.  This heuristic is trivial.
      """
    return 0


def manhattanHeuristic(state, goal):
    cost_sum = np.abs(goal[0] - state[0]) + np.abs(goal[1] - state[1])
    return cost_sum


def euclideanHeuristic(state, goal):
    cost_sum = np.sqrt((goal[0] - state[0]) ** 2 + (goal[1] - state[1]) ** 2)
    return cost_sum


def aStarSearch(xI, xG, n, m, O, heuristic='nullHeuristic'):
    if heuristic == 'manhattan':
        heuristic_func = manhattanHeuristic
    elif heuristic == 'euclidean':
        heuristic_func = euclideanHeuristic
    else:
        heuristic_func = nullHeuristic

    controls = ((1, 0), (-1, 0), (0, 1), (0, -1))
    pqueue = list()  # Create a priority queue
    actions = list()
    cost = getCostOfActions(xI, actions, O) + heuristic_func(xI, xG)
    best_cost_library = {xI: cost}
    pqueue.append((cost, xI))
    visited = set(xI)
    parents = {xI: None}
    control_library = {}

    while len(pqueue) > 0:
        pqueue.sort(key=lambda pair: pair[0])  # Sort the queue by the cost value
        pair = pqueue.pop(0)
        node = pair[1]  # Get the node with the smallest cost
        if node == xG:
            s = xG
            while parents[s] is not None:
                action = tuple(map(lambda i, j: i - j, s, parents[s]))
                s = parents[s]
                actions.insert(0, action)
            total_cost = getCostOfActions(xI, actions, O)  # Compute the total cost
            num_visited = len(visited)  # Obtain the number of visited nodes
            return actions, total_cost, num_visited
        else:
            for control in controls:  # For all inputs
                collided = collisionCheck(node, control, O)
                if not collided:  # The input is valid
                    child = tuple(map(lambda i, j: i + j, node, control))  # Obtain the new node
                    if child != parents[node]:  # Avoid the repeated path
                        cost_state = heuristic_func(child, xG) + getCostOfActions(node, [control], O)
                        control_library[control] = cost_state
                        cost = cost_state + best_cost_library[node]  # Compute the cost
                        if child not in visited:
                            best_cost_library[child] = cost  # Update its optimal cost
                            parents[child] = node  # Update its parent
                            visited.add(child)  # Mark it as visited
                            pqueue.append((control_library[control], child))
                    else:
                        continue
    return False


# Plots the path
def showPath(xI, xG, path, n, m, O):
    gridpath = makePath(xI, xG, path, n, m, O)
    fig, ax = plt.subplots(1, 1)  # make a figure + axes
    ax.imshow(gridpath)  # Plot it
    ax.invert_yaxis()  # Needed so that bottom left is (0,0)


if __name__ == '__main__':
    # Run test using smallMaze.py (loads n,m,O)
    from code.smallMaze import *

    # from mediumMaze import *  # try these mazes too
    # from bigMaze import *     # try these mazes too
    maze(n, m, O)  # prints the maze

    # Sample collision check
    x, u = (5, 4), (1, 0)
    testObs = [[6, 6, 4, 4]]
    collided = collisionCheck(x, u, testObs)
    print('Collision!' if collided else 'No collision!')

    # Sample path plotted to goal
    xI = (1, 1)
    xG = (20, 1)
    actions = [(1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (0, 1),
               (1, 0), (1, 0), (1, 0), (0, -1), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 0)]
    path = getPathFromActions(xI, actions)
    showPath(xI, xG, path, n, m, O)

    # Cost of that path with various cost functions
    simplecost = getCostOfActions(xI, actions, O)
    westcost = stayWestCost(xI, actions, O)
    eastcost = stayEastCost(xI, actions, O)
    print('Basic cost was %d, stay west cost was %d, stay east cost was %d' %
          (simplecost, westcost, eastcost))

    plt.show()
