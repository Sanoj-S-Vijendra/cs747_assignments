import argparse
import numpy as np
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, PULP_CBC_CMD

def get_mdp_datas(mdp):
    return mdp["numStates"], mdp["numActions"], mdp["endStates"].copy(), mdp["transitions"].copy(), mdp["gamma"]

def policy_evaluation(mdp_data, policy, V_init=None, iter=30):
    S,nA,end_states,transitions,gamma = get_mdp_datas(mdp_data)
    V = V_init if V_init is not None else np.zeros(S)
    for _ in range(iter):
        V_prev = np.copy(V)
        for s in range(S):
            if s in end_states:
                V[s] = 0.0
                continue
            action = policy[s]
            new_v = 0.0
            if transitions[s][action]:
                for s1, (r, p) in transitions[s][action].items():
                    new_v+=(p*(r+gamma*V_prev[s1]))
            V[s] = new_v
    return V

def howards_policy_iteration(mdp_data):
    S,A,end_states,transitions,gamma = get_mdp_datas(mdp_data)
    policy = np.zeros(S, dtype=int)
    improv = True
    V = None
    while improv:
        V = policy_evaluation(mdp_data, policy, V_init=V)
        improv = False
        for s in range(S):
            if s in end_states:
                continue
            old_action = policy[s]
            q_values = np.zeros(A)
            for a in range(A):
                q_s_a = 0
                if transitions[s][a]:
                    for s1,(r,p) in transitions[s][a].items():
                        q_s_a+=(p*(r+gamma*V[s1]))
                q_values[a]=q_s_a
            best_action = np.argmax(q_values)
            policy[s] = best_action
            if(old_action!=best_action):
                improv = True
    return V, policy

def linear_programming(mdp_data):
    S,A,end_states,transitions,gamma = get_mdp_datas(mdp_data)
    prob = LpProblem("MDP_LP", LpMinimize)
    V_vars = [LpVariable(f"V_{s}") for s in range(S)]
    prob += lpSum(V_vars),"sum_V"
    for s in range(S):
        if s in end_states:
            prob += (V_vars[s] == 0.0)
            continue
        for a in range(A):
            if transitions[s][a]:
                exp_r = sum(p*r for s1,(r, p) in transitions[s][a].items())
                new_value = lpSum([p*V_vars[s1] for s1, (_, p) in transitions[s][a].items()])
                prob += V_vars[s] >= exp_r + gamma*new_value
    prob.solve(PULP_CBC_CMD(msg=0))
    V_star = np.array([v.value() for v in V_vars])
    policy_star = np.zeros(S, dtype=int)
    for s in range(S):
        if s in end_states:
            continue
        q_values = np.zeros(A)
        for a in range(A):
            q_s_a = 0
            if transitions[s][a]:
                for s1,(r,p) in transitions[s][a].items():
                    q_s_a+=(p*(r+gamma*V_star[s1]))
            q_values[a] = q_s_a
        policy_star[s] = np.argmax(q_values)
    return V_star, policy_star

def args_parser():
    parser = argparse.ArgumentParser(description="MDP Planner")
    parser.add_argument("--mdp", required=True, help="Path to the MDP file.")
    parser.add_argument("--algorithm", default="lp", choices=["hpi", "lp"], help="Algorithm to use (hpi or lp). Default is hpi.")
    parser.add_argument("--policy", help="Path to a policy file to evaluate.")
    return parser.parse_args()

def load_data(filepath):
    mdp_data = {
        "numStates": 0,
        "numActions": 0,
        "endStates": [],
        "transitions": [],
        "mdptype": "",
        "gamma": 1.0
    }
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "numStates":
                mdp_data["numStates"] = int(parts[1])
            elif parts[0] == "numActions":
                mdp_data["numActions"] = int(parts[1])
            elif parts[0] == "end":
                if int(parts[1])!=-1:
                    mdp_data["endStates"] = [int(s) for s in parts[1:]]
            elif parts[0] == "transition":
                s1,ac,s2,r,p = int(parts[1]),int(parts[2]),int(parts[3]),float(parts[4]),float(parts[5])
                if not mdp_data["transitions"]:
                    S = mdp_data["numStates"]
                    A = mdp_data["numActions"]
                    mdp_data["transitions"] = [[{} for _ in range(A)] for _ in range(S)]
                mdp_data["transitions"][s1][ac][s2] = (r,p)
            elif parts[0] == "mdptype":
                mdp_data["mdptype"] = parts[1]
            elif parts[0] == "discount":
                mdp_data["gamma"] = float(parts[1])
    return mdp_data

def main():
    args = args_parser()
    mdp_data = load_data(args.mdp)
    V = None
    policy = None
    if args.policy:
        with open(args.policy, 'r') as f:
            policy = np.array([int(line.strip()) for line in f])
        V = policy_evaluation(mdp_data, policy)
    else:
        if args.algorithm == "hpi":
            V,policy = howards_policy_iteration(mdp_data)
        elif args.algorithm == "lp":
            V,policy = linear_programming(mdp_data)
    for s in range(mdp_data["numStates"]):
        val = V[s] if s not in mdp_data["endStates"] else 0.0
        act = policy[s] if s not in mdp_data["endStates"] else 0
        print(f"{val:.6f}\t{act}")

if __name__ == "__main__":
    main()