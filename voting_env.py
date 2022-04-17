from collections import OrderedDict
import matplotlib.pyplot as plt
from functools import reduce
import math
import random

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import time
import sys
import copy
from voting import *
from numpy.random import default_rng

    """
    Get difference between public majority opinion and malicious's private profile
    """
def diff_public_attacker(nissues, pvs, ppc, nvoters):
    public_agreement = []
    public_weight = np.sum(pvs, axis = 1)
    print(public_weight)
    diff = []
    for i in range(nissues):
        if public_weight[i]>= nvoters/2:
            result = True
        else:
            result = False
        if result != ppc[i]:
            diff.append((i, max(public_weight[i], nvoters-public_weight[i])))
    dtype = [('index', int), ('weight', int)]
    diff = np.asarray(diff, dtype = dtype)
    diff = np.flip(np.sort(diff, order = 'weight'))
    return diff


class Candidate:
    def __init__(self, nissues, pvs):
        self.rng = default_rng()
        self.nissues = nissues
        self.private_profile = rng.choice([True, False], size=nissues)
        self.honesty = 1
        self.change = 0
        self.set_public_profile(pvs)

class MaliciousCandidate(Candidate):
    """
    Defines a malicious (pandering) candidate
    All of the same attributes as Candidate, and additionally:
        change - the proportion of issues the candidate will change to match
                 public opinion
    """
    def get_change(results=None):
        # if there are no results, proportion of issues that will be changed is 0
        if results is None:
            self.change = self.rng.random()
        else:
            pass

    def set_public_profile(self, changelist):
        # get popular positions on issues
        # deviate from private profile based on parameter `change`
        # action of malicious candidate
        # one round: given from results of MIP
        # multiround: given from RL results
        self.public_profile = copy.deepcopy(self.private_profile)
        for i in changelist:
            self.public_profile[i] = not self.public_profile[i]



class HonestCandidate(Candidate):
    def set_public_profile(self, pvs):
        self.public_profile = self.private_profile



class RD_voting(gym.Env):

    def __init__(self):
        #initilize
        self.ncans = 10
        self.k = 5  # committee size
        self.nvoters = 100
        self.nissues = 30
        self.vr = ""
        self.max_rnd=100
        high = 1
        low = 0
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(2,), dtype = np.float32)
        #self.seed()

    # def seed(self, seed=None):
    #
    #
    def step(self, action):
        num_change = Int(action*len(self.diff))
        change_list = [self.diff[i][1] for i in range(num_change)]
        self.malicious.set_public_profile(change_list)
        self.pvc = get_preferences_on_candidates(self.pvs, ppc)
        self.committee = get_committee(pvc, self.vr, self.k)
        self.round_result = get_outcomes_rd(self.committee, self.nissues)
        agree = agreement(self.malicious.private_profile, self.round_result)
        in_committe = self.malicious in self.commitee
        reward = agree * 10 + in_committe *3
        self.round += 1

        #next round
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        for can in self.candidates:
            can.private_profile = rng.choice([True, False], size=self.nissues)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        return [len(self.diff), self.malicious.honesty]


    def reset(self):
        self.round = 0
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.candidates = [HonestCandidate(self.nissues) for _ in range(self.ncans - 1)]
        self.malicious = MaliciousCandidate(self.nissues)
        self.candidates.append(self.malicious)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)

        return [len(self.diff), self.malicious.honesty]


class FRD_voting(gym.Env):

    def __init__(self):
        self.ncans = 10
        self.k = 5  # committee size
        self.nvoters = 100
        self.nissues = 30
        self.vr = ""
        self.max_rnd=100
        high = 1
        low = 0
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(2,), dtype = np.float32)
        #self.seed()

    # def seed(self, seed=None):
    #
    #
    def step(self, action):
        num_change = Int(action*len(self.diff))
        change_list = [self.diff[i][1] for i in range(num_change)]
        self.malicious.set_public_profile(change_list)
        self.pvc = get_preferences_on_candidates(self.pvs, ppc)
        self.committee = get_committee(pvc, self.vr, self.k)
        self.round_result = get_outcomes_frd(self.committee, self.nissues, self.pvs)
        agree = agreement(self.malicious.private_profile, self.round_result)
        in_committe = self.malicious in self.commitee
        reward = agree * 10 + in_committe *3
        self.round += 1

        #next round
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        for can in self.candidates:
            can.private_profile = rng.choice([True, False], size=self.nissues)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        return [len(self.diff), self.malicious.honesty]


    def reset(self):
        self.round = 0
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.candidates = [HonestCandidate(self.nissues) for _ in range(self.ncans - 1)]
        self.malicious = MaliciousCandidate(self.nissues)
        self.candidates.append(self.malicious)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)

        return [len(self.diff), self.malicious.honesty]

# def main():
#     rng = default_rng()
#     pvs = rng.choice([False, True], size=(10, 10))
#     pps = rng.choice([False, True], size=(10, 10))
#
#     print(diff_public_attacker(10, pvs, pps[1], 10))


#main()
