"""
Task 3: Optimized KL-UCB Implementation

This file implements both standard and optimized KL-UCB algorithms for multi-armed bandits.
The optimized version aims to reduce computational overhead while maintaining good regret performance.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Base Algorithm Class ------------------

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# ------------------ KL-UCB utilities ------------------
## You can define other helper functions here if needed

# ------------------ Optimized KL-UCB Algorithm ------------------

class KL_UCB_Optimized(Algorithm):
    """
    Optimized KL-UCB algorithm that reduces computation while maintaining identical regret.
    This implements a batched KL-UCB with exponential+binary search for safe pulls of the current best arm.
    """
    ## You can define other functions also in the class if needed
    
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        self.t = 0
        self.counts = np.zeros(num_arms,dtype=int)
        self.sums = np.zeros(num_arms)
        self.c=1.2
        self.curr_arm = 0
        self.rem_pulls = 0
        self.epsilon=1e-15
        #END EDITING HERE

    def kl_div(self, p,q):
        p = max(self.epsilon,min(1-self.epsilon,p))
        q = max(self.epsilon,min(1-self.epsilon,q))
        if abs(p-q)<self.epsilon:
            return 0
        return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))

    def get_ucb(self,p, n):
        rhs = math.log(self.t) if self.t > 0 else 0
        if((self.t>1) and (self.c>0)):
            rhs+=self.c*math.log(math.log(self.t))
        target = rhs/n
        l,r=p,1.0
        if abs(p-1.0)<self.epsilon:
            return 1.0
        for _ in range(30):
            q_mid=(l+r)/2.0
            if self.kl_div(p,q_mid)<target:
                l=q_mid
            else:
                r=q_mid
        return (l+r)/2.0
    
    def give_pull(self):
        #START EDITING HERE
        if self.t<self.num_arms:
            return self.t
        if self.rem_pulls>0:
            self.rem_pulls-=1
            return self.curr_arm
        ucbs = np.zeros(self.num_arms)
        for arm in range(self.num_arms):
            if self.counts[arm]==0:
                ucbs[arm]=float('inf')
                continue
            p = self.sums[arm]/self.counts[arm]
            ucbs[arm] = self.get_ucb(p, self.counts[arm])
        best_arm=np.argmax(ucbs)
        self.curr_arm= best_arm
        ucbs_copy = np.copy(ucbs)
        ucbs_copy[best_arm] = -np.inf
        sec_best = np.argmax(ucbs_copy)
        sec_ucb = ucbs_copy[sec_best]
        p_best = self.sums[self.curr_arm]/self.counts[self.curr_arm]
        m=1
        if p_best<sec_ucb:
            kl = self.kl_div(p_best, sec_ucb)
            if kl>self.epsilon:
                rhs = math.log(self.t) if self.t > 0 else 0
                if((self.t > 1)and(self.c > 0)):
                    rhs+=self.c*math.log(math.log(self.t))
                N = rhs/kl
                m = math.floor(N)-self.counts[self.curr_arm]
                m = max(1,int(m))
        else:
            p_sec = self.sums[sec_best]/self.counts[sec_best]
            N_sec = self.counts[sec_best]
            ucb_best = ucbs[self.curr_arm]
            if (ucb_best<(1.0-self.epsilon)) and N_sec>0:
                kl_div_race = self.kl_div(p_sec,ucb_best)
                if kl_div_race>self.epsilon:
                    if N_sec*kl_div_race > 50:       # arbitrary value
                         m = self.horizon-self.t
                    else:
                        t_new = math.exp(N_sec*kl_div_race)
                        m = math.floor(t_new-self.t)
                        m = max(1, int(m))
        self.rem_pulls=min(m,self.horizon-self.t)
        self.rem_pulls-=1
        return self.curr_arm
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        self.t += 1
        self.counts[arm_index] += 1
        self.sums[arm_index] += reward
        #END EDITING HERE

# ------------------ Bonus KL-UCB Algorithm (Optional - 1 bonus mark) ------------------

class KL_UCB_Bonus(Algorithm):
    """
    BONUS ALGORITHM (Optional - 1 bonus mark)
    
    This algorithm must produce EXACTLY IDENTICAL regret trajectories to KL_UCB_Standard
    while achieving significant speedup. Students implementing this will earn 1 bonus mark.
    
    Requirements for bonus:
    - Must produce identical regret trajectories (checked with strict tolerance)
    - Must achieve specified speedup thresholds on bonus testcases
    - Must include detailed explanation in report
    """
    # You can define other functions also in the class if needed

    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # can initialize member variables here
        #START EDITING HERE
        #END EDITING HERE
    
    def give_pull(self):
        #START EDITING HERE
        pass
        #END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        #START EDITING HERE
        pass
        #END EDITING HERE
