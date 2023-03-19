#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:12:40 2022

@author: Tianmu Wang (tiw028@ucsd.edu)
"""

from mdps.gridWorld import *

from mdps.nextState import nextState
from mdps.smallGrid import smallGrid
from mdps.mediumGrid import mediumGrid
from mdps.testGrid import testGrid
from mdps.costFunction import getCost, getCostBridge
import numpy as np
import copy

"""
Implement your value iteration algorithm
"""

actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]


def state_add_action(state: list, action: tuple):
    result = [state[0] + action[0], state[1] + action[1]]
    return result


def compute_q_probs(current, action, next_state, eta):  # action: tuple, next_state: tuple):

    computed_action = tuple(map(lambda x, y: x - y, next_state, current))
    if computed_action == action:
        q_probs = 1 - eta
    else:
        if computed_action == (-1 * action[0], -1 * action[1]):
            q_probs = 0
        else:
            q_probs = eta / 2
    return q_probs


def compute_p_probs(current, action, next_state, next_states_list, p_dict, O, eta):
    q_prob = compute_q_probs(current, action, next_state, eta)
    if not isObstacle(next_state, O):
        next_states_list.append(next_state)
        p_dict[str(next_state)] = q_prob
    else:
        if current not in next_states_list:
            next_states_list.append(current)
        p_dict[str(current)] += q_prob
    return p_dict


def compute_q_value(state, action, temp_values, cost, gridname, O, eta, gamma):
    # Initialize the current probability
    p_dict = {str(state): 0}
    next_state_list = []
    q_value = 0

    next_state_expected = state_add_action(state, action)
    p_dict = compute_p_probs(state, action, next_state_expected, next_state_list, p_dict, O, eta)

    next_state_left = state_add_action(state, (action[1], action[0]))
    p_dict = compute_p_probs(state, action, next_state_left, next_state_list, p_dict, O, eta)

    next_state_right = state_add_action(state, (-1 * action[1], -1 * action[0]))
    p_dict = compute_p_probs(state, action, next_state_right, next_state_list, p_dict, O, eta)

    next_state_opposite = state_add_action(state, (-1 * action[0], -1 * action[1]))
    p_dict = compute_p_probs(state, action, next_state_opposite, next_state_list, p_dict, O, eta)

    for next_s in next_state_list:
        q_value += p_dict[str(next_s)] * (cost(next_s, gridname) + gamma * temp_values[next_s[0], next_s[1]])

    return q_value


def valueIteration(gamma, cost, eta, gridname):
    if gridname == 'small':
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    elif gridname == 'medium':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
    elif gridname == 'test':
        n, m, O, START, WINSTATE, DISTANTEXIT, LOSESTATE, LOSESTATES = testGrid()
    else:
        raise NameError("Unknown grid")

    if cost == 'cost':
        cost = getCost
    elif cost == 'bridge':
        cost = getCostBridge
    else:
        raise NameError('Unknown cost')

    error = 1e-3

    # Initialization
    values = np.ones((n, m))
    policy = np.zeros((n, m))
    iterations = 0

    while True:
        iterations += 1
        previous_values = values.copy()
        # For s in S
        for i in range(n):
            for j in range(m):
                current_state = [i, j]
                if not isObstacle(current_state, O):
                    q_value_list = []
                    # For a in A
                    for action in actions:
                        # Compute Q(s,a)
                        q_value = compute_q_value(current_state, action, previous_values, cost, gridname, O, eta, gamma)
                        q_value_list.append((action, q_value))
                    items = min(q_value_list, key=lambda item: item[1])
                    # V(s) = min(Q(s,a))
                    values[i, j] = items[1]
                    # pi(s) = argmin(Q(s,a))
                    policy[i, j] = actions.index(items[0])
                else:
                    continue
        if np.amax(np.abs(np.subtract(previous_values, values))) < error:
            return values, policy, iterations


"""
Implement your policy iteration algorithm
"""


def policyIteration(gamma, cost, eta, gridname):
    """
    (Offline) Policy iteration with a discount factor gamma and
    pre-defined cost functions.
    Output:
    values: Numpy array of (n,m) dimensions
    policy: Numpy array of (n,m) dimensions
    """
    if gridname == 'small':
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    elif gridname == 'medium':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
    elif gridname == 'test':
        n, m, O, START, WINSTATE, DISTANTEXIT, LOSESTATE, LOSESTATES = testGrid()
    else:
        raise NameError("Unknown grid")

    if cost == 'cost':
        cost = getCost
    elif cost == 'bridge':
        cost = getCostBridge
    else:
        raise NameError('Unknown cost')

    error = 1e-3
    iterations_ = 0

    # Initialization
    values_ = np.zeros((n, m))
    policy_ = np.zeros((n, m))
    # Set Labels
    Evaluation = True
    policy_improvement = False

    while True:
        while Evaluation:
            previous_values_ = values_.copy()
            for i in range(n):
                for j in range(m):
                    current_state = [i, j]
                    if not isObstacle(current_state, O):
                        action = actions[int(policy_[i, j])]
                        q_value = compute_q_value(current_state, action, previous_values_, cost, gridname, O, eta,
                                                  gamma)
                        values_[i, j] = q_value
                    else:
                        continue
            if np.amax(np.abs(np.subtract(previous_values_, values_))) < error:
                policy_improvement = True
                break
        while policy_improvement:
            previous_values_ = values_.copy()
            iterations_ += 1
            previous_policy_ = policy_.copy()
            for i in range(n):
                for j in range(m):
                    current_state = [i, j]
                    # Check whether it is an obstacle
                    if not isObstacle(current_state, O):
                        q_value_list = []
                        for action in actions:
                            q_value = compute_q_value(current_state, action, previous_values_, cost, gridname, O, eta,
                                                      gamma)
                            q_value_list.append((action, q_value))
                        items = min(q_value_list, key=lambda item: item[1])
                        values_[current_state[0], current_state[1]] = items[1]
                        policy_[current_state[0], current_state[1]] = actions.index(items[0])
                    else:  # If it is an obstable, then skip it.
                        continue
            if not (policy_ == previous_policy_).all():
                Evaluation = True
                break
            else:  # Policy = new Policy
                Evaluation = False
                policy_improvement = False
                break
        if not Evaluation and not policy_improvement:
            return values_, policy_, iterations_


def optimalValues(question):
    """
    Please input your values of gamma and eta
    for each assignment problem here.
    """
    if question == 'a':
        gamma = 0.9
        eta = 0.2
        return gamma, eta
    elif question == 'b':
        gamma = 0.9
        eta = 0.2
        return gamma, eta
    elif question == 'c':
        gamma = 0.9
        eta = 0.2
        return gamma, eta
    elif question == 'd1':
        gamma = 0.3
        eta = 0
        return gamma, eta
    elif question == 'd2':
        gamma = 0.3
        eta = 0.2
        return gamma, eta
    elif question == 'd3':
        gamma = 0.9
        eta = 0.3
        return gamma, eta
    elif question == 'd4':
        gamma = 0.7
        eta = 0.2
        return gamma, eta
    else:
        pass
    return 0


def showPath(xI, xG, path, n, m, O):
    gridpath = makePath(xI, xG, path, n, m, O)
    fig, ax = plt.subplots(1, 1)  # make a figure + axes
    ax.imshow(gridpath)  # Plot it
    ax.invert_yaxis()  # Needed so that bottom left is (0,0)


# Function to actually plot the cost-to-gos
def plotValues(values, xI, xG, n, m, O):
    gridvalues = makeValues(values, xI, xG, n, m, O)
    fig, ax = plt.subplots()  # make a figure + axes
    ax.imshow(gridvalues)  # Plot it
    ax.invert_yaxis()  # Needed so that bottom left is (0,0)


def showValues(n, m, values, O):
    string = '------'
    for i in range(0, n):
        string = string + '-----'
    for j in range(0, m):
        print(string)
        out = '| '
        for i in range(0, n):
            jind = m - j - 1  # Need to reverse index so bottom-left is (0,0)
            if isObstacle((i, jind), O):
                out += 'Obs' + ' | '
            else:
                out += str(values[i, jind]) + ' | '
        print(out)
    print(string)


if __name__ == '__main__':
    gridname = 'small'
    # gridname = 'medium'

    if gridname == 'small':
        n, m, O, START, WINSTATE, LOSESTATE = smallGrid()
    elif gridname == 'medium':
        n, m, O, START, DISTANTEXIT, CLOSEEXIT, LOSESTATES = mediumGrid()
    elif gridname == 'test':
        n, m, O, START, WINSTATE, DISTANTEXIT, LOSESTATE, LOSESTATES = testGrid()
    else:
        raise NameError("Unknown grid")

    """
    # Case 1:
    """
    cost = 'cost'
    gamma, eta = optimalValues('q')
    values, policy, iterations = valueIteration(gamma, cost, eta, gridname)
    # values, policy, iterations = policyIteration(gamma, cost, eta, gridname)
