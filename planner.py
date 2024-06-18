from argparse import ArgumentParser as argp
import numpy as np
from pulp import *

def polyevaluate(policy, numStates, transition, gamma, end_states):
    v_old, v_new = np.zeros((2,numStates))
    while True:
        for i in range(numStates):
            if i in end_states: continue
            v_new[i] = np.sum([k[2]*(k[1] + (gamma*v_old[k[0]])) for k in transition[i][policy[i]]])
        if np.allclose(v_old, v_new, 0.0, max_error, True): break
        v_old = np.copy(v_new)
    return v_new

def hpi(numStates, numActions, transition, gamma, end_states):
    policy = np.zeros(numStates, dtype=int)
    while True:
        v_new = polyevaluate(policy, numStates, transition, gamma, end_states)
        improving_actions = 0
        for i in range(numStates):
            for j in range(numActions):
                action_value = sum([k[2]*(k[1] + gamma*v_new[k[0]]) for k in transition[i][j]])
                if action_value > v_new[i] + max_error:
                    improving_actions += 1
                    policy[i] = j
                    break 
        if not improving_actions: break
    return v_new, policy   

def vi(numStates, numActions, transition, gamma, end_states):
    v_old, v_new = np.zeros((2, numStates))
    policy = np.zeros(numStates, dtype=int)
    while True:
        for i in range(numStates):
            v_optimal = 0
            if i in end_states: continue
            for j in range(numActions):
                v_new[i] = np.sum([k[2]*(k[1] + (gamma*v_old[k[0]])) for k in transition[i][j]])
                if v_new[i] > v_optimal:
                    v_optimal, policy[i] = v_new[i], j
            v_new[i] = v_optimal
        if np.allclose(v_old, v_new, 0.0, max_error, True): break
        v_old = np.copy(v_new)
    return v_new, policy  

def lp(numStates, numActions, transition, gamma, end_states):
    prob = LpProblem("mdp", LpMinimize)
    v = [LpVariable('v'+str(i)) for i in range(numStates)]
    prob += lpSum(v)
    for i in range(numStates):
        if i in end_states:
            prob += (v[i] == 0)
            continue
        for j in range(numActions): 
            prob += (v[i] >= lpSum([k[2]*(k[1] + gamma*v[k[0]]) for k in transition[i][j]]))
    prob.solve(PULP_CBC_CMD(msg=0))
    v_new = [value(v[i]) for i in range(numStates)]
    policy = np.zeros(numStates, dtype=int)
    for i in range(numStates):
        diff = np.inf
        for j in range(numActions):
            action_value = np.sum([k[2]*(k[1] + gamma*v_new[k[0]]) for k in transition[i][j]])
            if np.allclose(action_value, v_new[i], 0.0, diff, True):
                policy[i], diff = j, abs(action_value - v_new[i])                
    return v_new, policy    

parser = argp()
parser.add_argument("--mdp", type=str)
parser.add_argument("--algorithm", type=str, default='lp')
parser.add_argument("--policy", type=str)
args = parser.parse_args()

content = list(map(lambda x: x.strip().split(), open(args.mdp).readlines()))
numStates = int(content[0][1])
numActions = int(content[1][1])
end_states = [int(endls) for endls in content[2][1:]]
gamma = float(content[-1][1])
max_error = 1e-12

transition = [[[] for i in range(numActions)] for j in range(numStates)]
for line in content[3:-2]:
    line = list(map(float, line[1:]))
    state, action, prime = int(line[0]), int(line[1]), int(line[2])
    transition[state][action].append((prime, *line[3:]))

if args.policy:
    pie = list(map(lambda x: int(x.strip().split()[0]), open(args.policy).readlines()))
    value = polyevaluate(pie, numStates, transition, gamma, end_states)
else:
    value, pie = globals()[args.algorithm](numStates, numActions, transition, gamma, end_states)

for i in range(len(value)):
    print(f'{np.round(value[i], 7)}\t{pie[i]}')