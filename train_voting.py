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
from numpy.random import default_rng
from stable_baselines3 import SAC, PPO, TD3, DQN
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import pandas as pd

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


"""
Gets preferences of voters on candidates by ordering them based on their agreement.
Params:
    pvs - voter preference profile on issues (np array of size (num voters, num issues))
    ppc - public candidate preference profile (np array of size (num cans, num issues))
"""
def get_preferences_on_candidates_with_malicious(pvs, candidates):
    # the ith entry is the ith voters ranking of candidates, so
    # pvc[i][0] = (lowest agreement value, lowest agreement index) for voter i
    pvc = []
    # loop over voter preference profiles on votes
    for voter in pvs:
        # get the agreement of the voter with each candidate
        agreements = [(agreement(voter, candidates[i].public_profile) * candidates[i].honesty, i + 1) for i in range(len(candidates))]
        #print(agreements)
        # sort so that highest agreement is last
        agreements.sort()
        pvc.append(agreements)
        #print(pvc)
    return np.array(pvc)

def get_preferences_on_candidates_without_malicious(pvs, candidates):
    # the ith entry is the ith voters ranking of candidates, so
    # pvc[i][0] = (lowest agreement value, lowest agreement index) for voter i
    pvc = []
    # loop over voter preference profiles on votes
    for voter in pvs:
        # get the agreement of the voter with each candidate
        agreements = [(agreement(voter, candidates[i].private_profile) * candidates[i].honesty, i + 1) for i in range(len(candidates))]
        #print(agreements)
        # sort so that highest agreement is last
        agreements.sort()
        pvc.append(agreements)
        #print(pvc)
    return np.array(pvc)

def generate_preference(rng, num_voters, num_issues, p1, p2 = None):
    if p2 == None:
        return rng.choice([False, True], size=(num_voters, num_issues), p =[p1, 1-p1])
    else:
        first_group_num = num_voters//2
        second_group_num = num_voters - first_group_num
        first_group = rng.choice([False, True], size=(first_group_num, num_issues), p =[p1, 1-p1])
        second_group = rng.choice([False, True], size=(second_group_num, num_issues), p =[p2, 1-p2])
        return np.concatenate((first_group, second_group))


class Candidate:
    def __init__(self, nissues, pvs):
        self.rng = default_rng()
        self.nissues = nissues
        self.private_profile = self.rng.choice([True, False], size=nissues)
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

    def __init__(self, nissues, pvs, private = []):
        self.rng = default_rng()
        self.nissues = nissues
        if len(private) == 0:
            self.private_profile = self.rng.choice([True, False], size=nissues)
        else:
            self.private_profile = private
        self.honesty = 1
        self.change = 0

    def get_change(results=None):
        # if there are no results, proportion of issues that will be changed is 0
        if results is None:
            self.change = self.rng.random()
        else:
            pass

    def set_public_profile(self, action):
        # get popular positions on issues
        # deviate from private profile based on parameter `change`
        # action of malicious candidate
        # one round: given from results of MIP
        # multiround: given from RL results
        #print(changelist)
        self.public_profile = action
            #self.public_profile[i] = -self.public_profile[i]

class HonestCandidate(Candidate):
    def set_public_profile(self, pvs):
        self.public_profile = self.private_profile



class RD_voting_one_round(gym.Env):

    def __init__(self):
        #initilize
        self.rng = default_rng()
        self.ncans = 10
        self.k = 5  # committee size
        self.nvoters = 19
        self.nissues = 9
        #self.vr = weights
        self.vr = av
        self.max_rnd=5
        # high = 1
        # low = 0
        # self.action_space = spaces.Box(
        #     low=low,
        #     high=high,
        #     shape=(self.nissues,),
        #     dtype=np.float32
        # )
        self.action_space = spaces.MultiBinary(self.nissues)

        #self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(self.nissues,), dtype = np.float32)
        self.observation_space = spaces.MultiBinary(self.nvoters*self.nissues)
        #self.seed()

    def seed(self, seed=None):
        pass

    def step(self, action):
        #print(self.candidates[0].honesty)
        #num_change = int(action*len(self.diff))
        #print(self.diff)
        #change_list = [self.diff[i][0] for i in range(num_change)]
        true_action = []
        #print(action)
        for i in action:
            if i > 0:
                true_action.append(True)
            else:
                true_action.append(False)
        self.malicious.set_public_profile(true_action)
        self.ppc = []
        for can in self.candidates:
            self.ppc.append(can.public_profile)
        self.pvc = get_preferences_on_candidates_with_malicious(self.pvs, self.candidates)
        self.committee_index, malicious_score = get_committee(self.pvc, self.vr, self.k)
        #print(self.committee_index)
        self.committee = []
        for i in self.committee_index:
            #print(i)
            self.committee.append(self.candidates[i-1])
        #print(self.committee)
        self.round_result = get_outcomes_rd(self.committee, self.nissues, test = True)
        agree = agreement(self.malicious.private_profile, self.round_result)
        in_committee = self.malicious in self.committee
        #reward = in_committe * 1 + agree * 10
        # if in_committe:
        #     if num_change>0:
        #         reward = 1 + agree * 10
        #         self.chosen_1 += 1
        #     else:
        #         reward = agree * 3
        #     self.chosen+=1
        # else:
        #     self.malicious.honesty = min(self.malicious.honesty*1.01, 1)
        #reward = agree
        # if in_committe:
        #     reward = agree
        # else:
        #     reward = 0
        reward = malicious_score
        #if true_action != self.malicious.private_profile:
        for a,b in zip(true_action, self.malicious.private_profile):
            if a != b:
                self.chosen_1 += 1
                break
        self.round += 1
        #next round
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        for can in self.candidates:
            can.private_profile = self.rng.choice([True, False], size=self.nissues)
            if can != self.malicious:
                can.set_public_profile(self.pvs)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)

        if self.round >= self.max_rnd:
            done = True
            # print("Chosen as committee is ", str(self.chosen))
            # print("Chosen as committee by pandering is ", self.chosen_1)
            #print("Final honesty is ", str(self.malicious.honesty))
        else:
            done = False

        #public_weight = np.sum(self.pvs, axis = 0)
        pvs_vec = pvs_to_vector(self.pvs)


        return pvs_vec, reward, done, {}


def list_to_wol(list):
    result = ''
    result+='{'
    count = 1
    max_count = len(list)
    if len(np.shape(list)) == 1:
        result+=",".join([str(int(i)) for i in list])
        result+='}'
    else:
        for i in list:
            result+='{'
            result+=",".join([str(int(j)) for j in i])
            result+='}'
            if count < max_count:
                result+=','
                count+=1
        result+='}'
    return result

class RD_voting_mip_selfish(gym.Env):

    def __init__(self):
        #initilize
        self.rng = default_rng()
        self.ncans = 10
        self.k = 5  # committee size
        self.nvoters = 50
        self.nissues = 9
        #self.vr = weights
        self.vr = av
        self.max_rnd = 100
        # high = 1
        # low = 0
        # self.action_space = spaces.Box(
        #     low=low,
        #     high=high,
        #     shape=(self.nissues,),
        #     dtype=np.float32
        # )
        self.action_space = spaces.Discrete(self.nissues)

        #self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(self.nissues*self.nvoters+1,), dtype = np.float32)
        #self.observation_space = spaces.MultiBinary(self.nvoters*self.nissues + 1)
        #self.observation_space = spaces.Dict({'pvs': spaces.MultiBinary(self.nvoters*self.nissues), 'honesty': spaces.Box(low = -np.inf, high= np.inf, shape=(1,), dtype = np.float32), 'round': spaces.Discrete(self.max_rnd)})
        self.observation_space = spaces.Dict({'pvs': spaces.Box(low = 0, high = 1, shape=(self.nissues,), dtype = np.float32),
                                              'preference': spaces.MultiBinary(self.nissues),
                                              'honesty': spaces.Box(low = 0, high= 1, shape=(1,), dtype = np.float32),
                                              'round': spaces.Discrete(self.max_rnd+1)})
        #self.seed()
        self.session = WolframLanguageSession()
        self.session.evaluate('binVec = {(0 | 1) ..};')
        self.session.evaluate('distanceFunction[a : binVec][b : binVec] := HammingDistance[a, b];')
        self.session.evaluate('numApprovals[x : {binVec ..}, ys : binVec, k_] := Length[Select[Map[distanceFunction[ys], x], (# <= k) &]];')
        self.session.evaluate('maximizeApprovals[x : {binVec ..}, k_, penalty_ : (0 &)] := \
          Module[{y}, \
           With[{ys = Array[y, Length[First[x]]]}, \
            Maximize[{numApprovals[x, ys, k] + penalty[ys], \
               Thread[0 <= ys <= 1]}, \
              ys \[Element] Integers] // {First[#], ys /. Last[#]} &]];')

    def optimal_pander(self, profile, max_diff_for_approval, true_pref, max_issues_pandered):
        #version with constraint, via suitably large penalty
        self.session.evaluate('maximizeApprovals[x_, k_, ref_, j_] := \
          With[{p = -Length[x] - 1}, \
           maximizeApprovals[x, k, \
            If[distanceFunction[ref][#] <= j, 0, p] &]];')\

        result = self.session.evaluate(f'maximizeApprovals[{profile}, {max_diff_for_approval}, {true_pref}, {max_issues_pandered}]')
        #print(result)
        true_result = []
        for i in result[1]:
            true_result.append(i)
        return true_result


    def seed(self, seed=None):
        pass

    def step(self, action):
        #print(self.candidates[0].honesty)
        #num_change = int(action*len(self.diff))
        #print(self.diff)
        #change_list = [self.diff[i][0] for i in range(num_change)]
        #print(action)
        # print(list_to_wol(self.pvs))
        # print(list_to_wol(self.malicious.private_profile))
        malicious_profile = self.optimal_pander(list_to_wol(self.pvs), 4, list_to_wol(self.malicious.private_profile), int(action))
        true_action = []
        for i in malicious_profile:
            if i > 0:
                true_action.append(True)
            else:
                true_action.append(False)
        self.malicious.set_public_profile(true_action)
        self.ppc = []
        for can in self.candidates:
            self.ppc.append(can.public_profile)
        self.pvc = get_preferences_on_candidates_with_malicious(self.pvs, self.candidates)
        self.committee_index, malicious_score = get_committee(self.pvc, self.vr, self.k)
        #print(self.committee_index)
        self.committee = []
        for i in self.committee_index:
            #print(i)
            self.committee.append(self.candidates[i-1])
        #print(self.committee)
        pvc_o = get_preferences_on_candidates_without_malicious(self.pvs, self.candidates)
        committee_index_o, _ = get_committee(pvc_o, self.vr, self.k)
        committee_o = []
        for i in committee_index_o:
            committee_o.append(self.candidates[i-1])
        in_committee_o = self.malicious in committee_o
        self.round_result_o = get_outcomes_rd(committee_o, self.nissues, test = True)
        # difference_o = 0
        # for i, j in zip(self.voter_majority, self.round_result_o):
        #     if i!=j:
        #         difference_o+=1

        self.round_result = get_outcomes_rd(self.committee, self.nissues)
        agree = agreement(self.malicious.private_profile, self.round_result)
        in_committee = self.malicious in self.committee

        # difference = 0
        # for i, j in zip(self.voter_majority, self.round_result):
        #     if i!=j:
        #         difference+=1

        if in_committee:
            if not in_committee_o:
                reward = agree
            else:
                reward = 0
            self.chosen+=1
            #continue
        else:
            self.malicious.honesty = min(self.malicious.honesty*1.01, 1)
            reward = 0
        #reward = difference - difference_o
        # if in_committe:
        #     reward = agree
        # else:
        #     reward = 0
        #reward = malicious_score
        #if true_action != self.malicious.private_profile:
        for a,b in zip(true_action, self.malicious.private_profile):
            if a != b:
                self.chosen_1 += 1
                break
        self.round += 1
        #next round
        #self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        #Polarized
        #self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.25, 0.75)
        #Similar
        self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.5)
        for can in self.candidates:
            can.private_profile = self.rng.choice([False, True], size=self.nissues, p = [0.5,0.5])
            if can != self.malicious:
                can.set_public_profile(self.pvs)
        #self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        if self.round >= self.max_rnd:
            done = True
            # print("Chosen as committee is ", str(self.chosen))
            # print("Chosen as committee by pandering is ", self.chosen_1)
            #print("Final honesty is ", str(self.malicious.honesty))
        else:
            done = False

        public_weight = np.sum(self.pvs, axis = 0)/self.nvoters
        # self.voter_majority = []
        # self.voter_minority = []
        # for i in public_weight:
        #     if i > 0.5:
        #         self.voter_majority.append(True)
        #         self.voter_minority.append(False)
        #     else:
        #         self.voter_majority.append(False)
        #         self.voter_minority.append(True)
        # self.malicious.private_profile = self.voter_minority
        #pvs_vec = pvs_to_vector(self.pvs)
        state = {'pvs': public_weight,'preference':self.malicious.private_profile, 'honesty': self.malicious.honesty, 'round': self.round}


        return state, reward, done, {}


    def reset(self):
        self.round = 0
        self.chosen = 0
        self.chosen_1 = 0
        #self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        #polarized
        #self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.25, 0.75)
        #Similar
        self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.5)
        self.candidates = [HonestCandidate(self.nissues, self.pvs) for _ in range(self.ncans - 1)]
        self.malicious = MaliciousCandidate(self.nissues, self.pvs)
        self.candidates.append(self.malicious)
        for can in self.candidates:
            can.private_profile = self.rng.choice([False, True], size=self.nissues, p = [0.5,0.5])
            if can != self.malicious:
                can.set_public_profile(self.pvs)
        #self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        #print(self.diff)
        #print(self.pvs)
        public_weight = np.sum(self.pvs, axis = 0)/self.nvoters
        # self.voter_majority = []
        # self.voter_minority = []
        # for i in public_weight:
        #     if i > 0.5:
        #         self.voter_majority.append(True)
        #         self.voter_minority.append(False)
        #     else:
        #         self.voter_majority.append(False)
        #         self.voter_minority.append(True)
        #print(public_weight)
        # self.malicious.private_profile = self.voter_minority
        #pvs_vec = pvs_to_vector(self.pvs)
        state = {'pvs': public_weight,'preference': self.malicious.private_profile, 'honesty': self.malicious.honesty, 'round': 0}
        return state


class FRD_voting_mip_selfish(gym.Env):

    def __init__(self):
        #initilize
        self.rng = default_rng()
        self.ncans = 10
        self.k = 5  # committee size
        self.nvoters = 50
        self.nissues = 9
        #self.vr = weights
        self.vr = av
        self.max_rnd = 100
        # high = 1
        # low = 0
        # self.action_space = spaces.Box(
        #     low=low,
        #     high=high,
        #     shape=(self.nissues,),
        #     dtype=np.float32
        # )
        self.action_space = spaces.Discrete(self.nissues)

        #self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(self.nissues*self.nvoters+1,), dtype = np.float32)
        #self.observation_space = spaces.MultiBinary(self.nvoters*self.nissues + 1)
        #self.observation_space = spaces.Dict({'pvs': spaces.MultiBinary(self.nvoters*self.nissues), 'honesty': spaces.Box(low = -np.inf, high= np.inf, shape=(1,), dtype = np.float32), 'round': spaces.Discrete(self.max_rnd)})
        self.observation_space = spaces.Dict({'pvs': spaces.Box(low = 0, high = 1, shape=(self.nissues,), dtype = np.float32),
                                              'preference': spaces.MultiBinary(self.nissues),
                                              'honesty': spaces.Box(low = 0, high= 1, shape=(1,), dtype = np.float32),
                                              'round': spaces.Discrete(self.max_rnd+1)})
        #self.seed()
        self.session = WolframLanguageSession()
        self.session.evaluate('binVec = {(0 | 1) ..};')
        self.session.evaluate('distanceFunction[a : binVec][b : binVec] := HammingDistance[a, b];')
        self.session.evaluate('numApprovals[x : {binVec ..}, ys : binVec, k_] := Length[Select[Map[distanceFunction[ys], x], (# <= k) &]];')
        self.session.evaluate('maximizeApprovals[x : {binVec ..}, k_, penalty_ : (0 &)] := \
          Module[{y}, \
           With[{ys = Array[y, Length[First[x]]]}, \
            Maximize[{numApprovals[x, ys, k] + penalty[ys], \
               Thread[0 <= ys <= 1]}, \
              ys \[Element] Integers] // {First[#], ys /. Last[#]} &]];')

    def optimal_pander(self, profile, max_diff_for_approval, true_pref, max_issues_pandered):
        #version with constraint, via suitably large penalty
        self.session.evaluate('maximizeApprovals[x_, k_, ref_, j_] := \
          With[{p = -Length[x] - 1}, \
           maximizeApprovals[x, k, \
            If[distanceFunction[ref][#] <= j, 0, p] &]];')\

        result = self.session.evaluate(f'maximizeApprovals[{profile}, {max_diff_for_approval}, {true_pref}, {max_issues_pandered}]')
        #print(result)
        true_result = []
        for i in result[1]:
            true_result.append(i)
        return true_result


    def seed(self, seed=None):
        pass

    def step(self, action):
        #print(self.candidates[0].honesty)
        #num_change = int(action*len(self.diff))
        #print(self.diff)
        #change_list = [self.diff[i][0] for i in range(num_change)]
        #print(action)
        # print(list_to_wol(self.pvs))
        # print(list_to_wol(self.malicious.private_profile))
        malicious_profile = self.optimal_pander(list_to_wol(self.pvs), 4, list_to_wol(self.malicious.private_profile), int(action))
        true_action = []
        for i in malicious_profile:
            if i > 0:
                true_action.append(True)
            else:
                true_action.append(False)
        self.malicious.set_public_profile(true_action)
        self.ppc = []
        for can in self.candidates:
            self.ppc.append(can.public_profile)
        self.pvc = get_preferences_on_candidates_with_malicious(self.pvs, self.candidates)
        self.committee_index, malicious_score = get_committee(self.pvc, self.vr, self.k)
        #print(self.committee_index)
        self.committee = []
        for i in self.committee_index:
            #print(i)
            self.committee.append(self.candidates[i-1])
        #print(self.committee)
        pvc_o = get_preferences_on_candidates_without_malicious(self.pvs, self.candidates)
        committee_index_o, _ = get_committee(pvc_o, self.vr, self.k)
        committee_o = []
        for i in committee_index_o:
            committee_o.append(self.candidates[i-1])
        in_committee_o = self.malicious in committee_o
        self.round_result_o = get_outcomes_frd_deterministic(committee_o, self.nissues, self.pvs, 0.5, test = True)
        # difference_o = 0
        # for i, j in zip(self.voter_majority, self.round_result_o):
        #     if i!=j:
        #         difference_o+=1

        self.round_result = get_outcomes_frd_deterministic(self.committee, self.nissues, self.pvs, 0.5)
        agree = agreement(self.malicious.private_profile, self.round_result)
        in_committee = self.malicious in self.committee

        # difference = 0
        # for i, j in zip(self.voter_majority, self.round_result):
        #     if i!=j:
        #         difference+=1

        if in_committee:
            if not in_committee_o:
                reward = agree
            else:
                reward = 0
            self.chosen+=1
            #continue
        else:
            self.malicious.honesty = min(self.malicious.honesty*1.01, 1)
            reward = 0
        #reward = difference - difference_o
        # if in_committe:
        #     reward = agree
        # else:
        #     reward = 0
        #reward = malicious_score
        #if true_action != self.malicious.private_profile:
        for a,b in zip(true_action, self.malicious.private_profile):
            if a != b:
                self.chosen_1 += 1
                break
        self.round += 1
        #next round
        #self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        #Polarized
        #self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.25, 0.75)
        #Similar
        self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.5)
        for can in self.candidates:
            can.private_profile = self.rng.choice([False, True], size=self.nissues, p = [0.5,0.5])
            if can != self.malicious:
                can.set_public_profile(self.pvs)
        #self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        if self.round >= self.max_rnd:
            done = True
            # print("Chosen as committee is ", str(self.chosen))
            # print("Chosen as committee by pandering is ", self.chosen_1)
            #print("Final honesty is ", str(self.malicious.honesty))
        else:
            done = False

        public_weight = np.sum(self.pvs, axis = 0)/self.nvoters
        # self.voter_majority = []
        # self.voter_minority = []
        # for i in public_weight:
        #     if i > 0.5:
        #         self.voter_majority.append(True)
        #         self.voter_minority.append(False)
        #     else:
        #         self.voter_majority.append(False)
        #         self.voter_minority.append(True)
        # self.malicious.private_profile = self.voter_minority
        #pvs_vec = pvs_to_vector(self.pvs)
        state = {'pvs': public_weight,'preference':self.malicious.private_profile, 'honesty': self.malicious.honesty, 'round': self.round}


        return state, reward, done, {}


    def reset(self):
        self.round = 0
        self.chosen = 0
        self.chosen_1 = 0
        #Polarized
        #self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.25, 0.75)
        #Similar
        self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.5)
        self.candidates = [HonestCandidate(self.nissues, self.pvs) for _ in range(self.ncans - 1)]
        self.malicious = MaliciousCandidate(self.nissues, self.pvs)
        self.candidates.append(self.malicious)
        for can in self.candidates:
            can.private_profile = self.rng.choice([False, True], size=self.nissues, p = [0.5,0.5])
            if can != self.malicious:
                can.set_public_profile(self.pvs)
        #self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        #print(self.diff)
        #print(self.pvs)
        public_weight = np.sum(self.pvs, axis = 0)/self.nvoters
        # self.voter_majority = []
        # self.voter_minority = []
        # for i in public_weight:
        #     if i > 0.5:
        #         self.voter_majority.append(True)
        #         self.voter_minority.append(False)
        #     else:
        #         self.voter_majority.append(False)
        #         self.voter_minority.append(True)
        #print(public_weight)
        # self.malicious.private_profile = self.voter_minority
        #pvs_vec = pvs_to_vector(self.pvs)
        state = {'pvs': public_weight,'preference': self.malicious.private_profile, 'honesty': self.malicious.honesty, 'round': 0}
        return state



class RD_voting_mip_malicious(gym.Env):

    def __init__(self, num):
        #initilize
        self.rng = default_rng()
        self.ncans = 10
        self.k = 5  # committee size
        self.nvoters = 50
        self.nissues = 9
        #self.vr = weights
        self.vr = av
        self.max_rnd = 100
        self.num_strategic = num
        self.selfish = False
        # high = 1
        # low = 0
        # self.action_space = spaces.Box(
        #     low=low,
        #     high=high,
        #     shape=(self.nissues,),
        #     dtype=np.float32
        # )
        self.action_space = spaces.Discrete((self.nissues+1)**self.num_strategic)
        #self.action_space = spaces.MultiDiscrete([self.nissues for _ in range(self.num_strategic)])

        #self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(self.nissues*self.nvoters+1,), dtype = np.float32)
        #self.observation_space = spaces.MultiBinary(self.nvoters*self.nissues + 1)
        #self.observation_space = spaces.Dict({'pvs': spaces.MultiBinary(self.nvoters*self.nissues), 'honesty': spaces.Box(low = -np.inf, high= np.inf, shape=(1,), dtype = np.float32), 'round': spaces.Discrete(self.max_rnd)})
        self.observation_space = spaces.Dict({'pvs': spaces.Box(low = 0, high = 1, shape=(self.nissues,), dtype = np.float32),
                                              'preference': spaces.MultiBinary(self.nissues),
                                              'honesty': spaces.Box(low = 0, high= 1, shape=(self.num_strategic,), dtype = np.float32),
                                              'round': spaces.Discrete(self.max_rnd+1)})
        #self.seed()
        self.session = WolframLanguageSession()
        self.session.evaluate('binVec = {(0 | 1) ..};')
        self.session.evaluate('distanceFunction[a : binVec][b : binVec] := HammingDistance[a, b];')
        self.session.evaluate('numApprovals[x : {binVec ..}, ys : binVec, k_] := Length[Select[Map[distanceFunction[ys], x], (# <= k) &]];')
        self.session.evaluate('maximizeApprovals[x : {binVec ..}, k_, penalty_ : (0 &)] := \
          Module[{y}, \
           With[{ys = Array[y, Length[First[x]]]}, \
            Maximize[{numApprovals[x, ys, k] + penalty[ys], \
               Thread[0 <= ys <= 1]}, \
              ys \[Element] Integers] // {First[#], ys /. Last[#]} &]];')

    def optimal_pander(self, profile, max_diff_for_approval, true_pref, max_issues_pandered):
        #version with constraint, via suitably large penalty
        self.session.evaluate('maximizeApprovals[x_, k_, ref_, j_] := \
          With[{p = -Length[x] - 1}, \
           maximizeApprovals[x, k, \
            If[distanceFunction[ref][#] <= j, 0, p] &]];')\

        result = self.session.evaluate(f'maximizeApprovals[{profile}, {max_diff_for_approval}, {true_pref}, {max_issues_pandered}]')
        #print(result)
        true_result = []
        for i in result[1]:
            true_result.append(i)
        return true_result


    def seed(self, seed=None):
        pass

    def step(self, action):
        #print(self.candidates[0].honesty)
        #num_change = int(action*len(self.diff))
        #print(self.diff)
        #change_list = [self.diff[i][0] for i in range(num_change)]
        #print(action)
        # print(list_to_wol(self.pvs))
        # print(list_to_wol(self.malicious.private_profile))
        true_num = reindex(self.nissues, self.num_strategic, action)
        for agent, i in zip(self.malicious, true_num):
            malicious_profile = self.optimal_pander(list_to_wol(self.pvs), 4, list_to_wol(agent.private_profile), int(i))
            true_action = []
            for i in malicious_profile:
                if i > 0:
                    true_action.append(True)
                else:
                    true_action.append(False)
            agent.set_public_profile(true_action)
        self.ppc = []
        for can in self.candidates:
            self.ppc.append(can.public_profile)
        self.pvc = get_preferences_on_candidates_with_malicious(self.pvs, self.candidates)
        self.committee_index, malicious_score = get_committee(self.pvc, self.vr, self.k)
        #print(self.committee_index)
        self.committee = []
        for i in self.committee_index:
            #print(i)
            self.committee.append(self.candidates[i-1])
        #print(self.committee)
        pvc_o = get_preferences_on_candidates_without_malicious(self.pvs, self.candidates)
        committee_index_o, _ = get_committee(pvc_o, self.vr, self.k)
        committee_o = []
        for i in committee_index_o:
            committee_o.append(self.candidates[i-1])
        in_committee_o = False
        for agent in self.malicious:
            if agent in committee_o:
                in_committe_o = True
                break
        self.round_result_o = get_outcomes_rd(committee_o, self.nissues, test = True)
        difference_o = 0
        for i, j in zip(self.voter_majority, self.round_result_o):
            if i!=j:
                difference_o+=1

        self.round_result = get_outcomes_rd(self.committee, self.nissues)
        #agree = agreement(self.malicious.private_profile, self.round_result)
        in_committee = False
        for agent in self.malicious:
            if agent in self.committee:
                in_committee = True
                break

        difference = 0
        for i, j in zip(self.voter_majority, self.round_result):
            if i!=j:
                difference+=1

        if in_committee:
            # if not in_committee_o:
            #     reward = agree * 10
            # else:
            #     reward = agree
            self.chosen+=1
            #continue
        for agent in self.malicious:
            if agent not in self.committee:
                agent.honesty = min(agent.honesty*1.01, 1)
            #reward = 0
        reward = difference - difference_o
        # if in_committe:
        #     reward = agree
        # else:
        #     reward = 0
        #reward = malicious_score
        #if true_action != self.malicious.private_profile:
        # for a,b in zip(true_action, self.malicious.private_profile):
        #     if a != b:
        #         self.chosen_1 += 1
        #         break
        self.round += 1
        #next round
        #self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.25)
        for can in self.candidates:
            can.private_profile = self.rng.choice([False, True], size=self.nissues, p = [0.75,0.25])
            if can not in self.malicious:
                can.set_public_profile(self.pvs)
        #self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        if self.round >= self.max_rnd:
            done = True
            # print("Chosen as committee is ", str(self.chosen))
            # print("Chosen as committee by pandering is ", self.chosen_1)
            #print("Final honesty is ", str(self.malicious.honesty))
        else:
            done = False

        public_weight = np.sum(self.pvs, axis = 0)/self.nvoters
        self.voter_majority = []
        self.voter_minority = []
        for i in public_weight:
            if i > 0.5:
                self.voter_majority.append(True)
                self.voter_minority.append(False)
            else:
                self.voter_majority.append(False)
                self.voter_minority.append(True)
        #print(public_weight)
        for agent in self.malicious:
            agent.private_profile = self.voter_minority

        #pvs_vec = pvs_to_vector(self.pvs)
        honesty = []
        for agent in self.malicious:
            honesty.append(agent.honesty)
        state = {'pvs': public_weight,'preference':self.voter_minority, 'honesty': honesty, 'round': self.round}


        return state, reward, done, {}


    def reset(self):
        self.round = 0
        self.chosen = 0
        self.chosen_1 = 0
        #self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.25)
        self.candidates = [HonestCandidate(self.nissues, self.pvs) for _ in range(self.ncans - self.num_strategic)]
        self.malicious = []
        for _ in range(self.num_strategic):
            self.malicious.append(MaliciousCandidate(self.nissues, self.pvs))
        for agent in self.malicious:
            self.candidates.append(agent)
        for can in self.candidates:
            can.private_profile = self.rng.choice([False, True], size=self.nissues, p = [0.75,0.25])
            if can not in self.malicious:
                can.set_public_profile(self.pvs)
        #self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        #print(self.diff)
        #print(self.pvs)
        public_weight = np.sum(self.pvs, axis = 0)/self.nvoters
        self.voter_majority = []
        self.voter_minority = []
        for i in public_weight:
            if i > 0.5:
                self.voter_majority.append(True)
                self.voter_minority.append(False)
            else:
                self.voter_majority.append(False)
                self.voter_minority.append(True)
        #print(public_weight)
        for agent in self.malicious:
            agent.private_profile = self.voter_minority
        #print(public_weight)
        #pvs_vec = pvs_to_vector(self.pvs)
        honesty = []
        for agent in self.malicious:
            honesty.append(agent.honesty)
        state = {'pvs': public_weight,'preference': self.voter_minority, 'honesty': honesty, 'round': 0}
        return state


class FRD_voting_mip_malicious(gym.Env):

    def __init__(self, num):
        #initilize
        self.rng = default_rng()
        self.ncans = 10
        self.k = 5  # committee size
        self.nvoters = 50
        self.nissues = 9
        #self.vr = weights
        self.vr = av
        self.max_rnd = 100
        self.num_strategic = num
        self.selfish = False
        # high = 1
        # low = 0
        # self.action_space = spaces.Box(
        #     low=low,
        #     high=high,
        #     shape=(self.nissues,),
        #     dtype=np.float32
        # )
        #self.action_space = spaces.Discrete(self.nissues)
        #self.action_space = spaces.MultiDiscrete([self.nissues for _ in range(self.num_strategic)])
        self.action_space = spaces.Discrete((self.nissues+1)**self.num_strategic)

        #self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(self.nissues*self.nvoters+1,), dtype = np.float32)
        #self.observation_space = spaces.MultiBinary(self.nvoters*self.nissues + 1)
        #self.observation_space = spaces.Dict({'pvs': spaces.MultiBinary(self.nvoters*self.nissues), 'honesty': spaces.Box(low = -np.inf, high= np.inf, shape=(1,), dtype = np.float32), 'round': spaces.Discrete(self.max_rnd)})
        self.observation_space = spaces.Dict({'pvs': spaces.Box(low = 0, high = 1, shape=(self.nissues,), dtype = np.float32),
                                              'preference': spaces.MultiBinary(self.nissues),
                                              'honesty': spaces.Box(low = 0, high= 1, shape=(self.num_strategic,), dtype = np.float32),
                                              'round': spaces.Discrete(self.max_rnd+1)})
        #self.seed()
        self.session = WolframLanguageSession()
        self.session.evaluate('binVec = {(0 | 1) ..};')
        self.session.evaluate('distanceFunction[a : binVec][b : binVec] := HammingDistance[a, b];')
        self.session.evaluate('numApprovals[x : {binVec ..}, ys : binVec, k_] := Length[Select[Map[distanceFunction[ys], x], (# <= k) &]];')
        self.session.evaluate('maximizeApprovals[x : {binVec ..}, k_, penalty_ : (0 &)] := \
          Module[{y}, \
           With[{ys = Array[y, Length[First[x]]]}, \
            Maximize[{numApprovals[x, ys, k] + penalty[ys], \
               Thread[0 <= ys <= 1]}, \
              ys \[Element] Integers] // {First[#], ys /. Last[#]} &]];')

    def optimal_pander(self, profile, max_diff_for_approval, true_pref, max_issues_pandered):
        #version with constraint, via suitably large penalty
        self.session.evaluate('maximizeApprovals[x_, k_, ref_, j_] := \
          With[{p = -Length[x] - 1}, \
           maximizeApprovals[x, k, \
            If[distanceFunction[ref][#] <= j, 0, p] &]];')\

        result = self.session.evaluate(f'maximizeApprovals[{profile}, {max_diff_for_approval}, {true_pref}, {max_issues_pandered}]')
        #print(result)
        true_result = []
        for i in result[1]:
            true_result.append(i)
        return true_result


    def seed(self, seed=None):
        pass

    def step(self, action):
        #print(self.candidates[0].honesty)
        #num_change = int(action*len(self.diff))
        #print(self.diff)
        #change_list = [self.diff[i][0] for i in range(num_change)]
        #print(action)
        # print(list_to_wol(self.pvs))
        # print(list_to_wol(self.malicious.private_profile))
        true_num = reindex(self.nissues, self.num_strategic, action)
        for agent, i in zip(self.malicious, true_num):
            malicious_profile = self.optimal_pander(list_to_wol(self.pvs), 4, list_to_wol(agent.private_profile), int(i))
            true_action = []
            for i in malicious_profile:
                if i > 0:
                    true_action.append(True)
                else:
                    true_action.append(False)
            agent.set_public_profile(true_action)
        self.ppc = []
        for can in self.candidates:
            self.ppc.append(can.public_profile)
        self.pvc = get_preferences_on_candidates_with_malicious(self.pvs, self.candidates)
        self.committee_index, malicious_score = get_committee(self.pvc, self.vr, self.k)
        #print(self.committee_index)
        self.committee = []
        for i in self.committee_index:
            #print(i)
            self.committee.append(self.candidates[i-1])
        #print(self.committee)
        pvc_o = get_preferences_on_candidates_without_malicious(self.pvs, self.candidates)
        committee_index_o, _ = get_committee(pvc_o, self.vr, self.k)
        committee_o = []
        for i in committee_index_o:
            committee_o.append(self.candidates[i-1])
        in_committee_o = False
        for agent in self.malicious:
            if agent in committee_o:
                in_committe_o = True
                break
        self.round_result_o = get_outcomes_frd_deterministic(committee_o, self.nissues, self.pvs, 0.5, test = True)
        difference_o = 0
        for i, j in zip(self.voter_majority, self.round_result_o):
            if i!=j:
                difference_o+=1

        self.round_result = get_outcomes_frd_deterministic(self.committee, self.nissues, self.pvs, 0.5)
        #agree = agreement(self.malicious.private_profile, self.round_result)
        in_committee = False
        for agent in self.malicious:
            if agent in self.committee:
                in_committee = True
                break

        difference = 0
        for i, j in zip(self.voter_majority, self.round_result):
            if i!=j:
                difference+=1

        if in_committee:
            # if not in_committee_o:
            #     reward = agree * 10
            # else:
            #     reward = agree
            self.chosen+=1
            #continue
        for agent in self.malicious:
            if agent not in self.committee:
                agent.honesty = min(agent.honesty*1.01, 1)
            #reward = 0
        reward = difference - difference_o
        # if in_committe:
        #     reward = agree
        # else:
        #     reward = 0
        #reward = malicious_score
        #if true_action != self.malicious.private_profile:
        # for a,b in zip(true_action, self.malicious.private_profile):
        #     if a != b:
        #         self.chosen_1 += 1
        #         break
        self.round += 1
        #next round
        #self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.25)
        for can in self.candidates:
            can.private_profile = self.rng.choice([False, True], size=self.nissues, p = [0.25,0.75])
            if can not in self.malicious:
                can.set_public_profile(self.pvs)
        #self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        if self.round >= self.max_rnd:
            done = True
            # print("Chosen as committee is ", str(self.chosen))
            # print("Chosen as committee by pandering is ", self.chosen_1)
            #print("Final honesty is ", str(self.malicious.honesty))
        else:
            done = False

        public_weight = np.sum(self.pvs, axis = 0)/self.nvoters
        self.voter_majority = []
        self.voter_minority = []
        for i in public_weight:
            if i > 0.5:
                self.voter_majority.append(True)
                self.voter_minority.append(False)
            else:
                self.voter_majority.append(False)
                self.voter_minority.append(True)
        #print(public_weight)
        for agent in self.malicious:
            agent.private_profile = self.voter_minority

        #pvs_vec = pvs_to_vector(self.pvs)
        honesty = []
        for agent in self.malicious:
            honesty.append(agent.honesty)
        state = {'pvs': public_weight,'preference':self.voter_minority, 'honesty': honesty, 'round': self.round}
        return state, reward, done, {}


    def reset(self):
        self.round = 0
        self.chosen = 0
        self.chosen_1 = 0
        #self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.pvs = generate_preference(self.rng, self.nvoters, self.nissues, 0.25)
        self.candidates = [HonestCandidate(self.nissues, self.pvs) for _ in range(self.ncans - self.num_strategic)]
        self.malicious = []
        for _ in range(self.num_strategic):
            self.malicious.append(MaliciousCandidate(self.nissues, self.pvs))
        for agent in self.malicious:
            self.candidates.append(agent)
        for can in self.candidates:
            can.private_profile = self.rng.choice([False, True], size=self.nissues, p = [0.25,0.75])
            if can not in self.malicious:
                can.set_public_profile(self.pvs)
        #self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        #print(self.diff)
        #print(self.pvs)
        public_weight = np.sum(self.pvs, axis = 0)/self.nvoters
        self.voter_majority = []
        self.voter_minority = []
        for i in public_weight:
            if i > 0.5:
                self.voter_majority.append(True)
                self.voter_minority.append(False)
            else:
                self.voter_majority.append(False)
                self.voter_minority.append(True)
        #print(public_weight)
        for agent in self.malicious:
            agent.private_profile = self.voter_minority
        #print(public_weight)
        #pvs_vec = pvs_to_vector(self.pvs)
        honesty = []
        for agent in self.malicious:
            honesty.append(agent.honesty)
        state = {'pvs': public_weight,'preference': self.voter_minority, 'honesty': honesty, 'round': 0}
        return state

if __name__ == '__main__':

    #env = FRD_voting_mip_selfish()
    #env = FRD_voting_mip_selfish()
    env = RD_voting_mip_malicious(1)
    #check_env(env)
    #vec_env = DummyVecEnv([lambda: FL_mnist()])
    #vec_env = VecCheckNan(vec_env, raise_exception = True)
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="RD_MIP_malicious_1_DQN_0.95_dissimilar_lr_1e-4_true",
                                             name_prefix='rl_model')
    #model = TD3("MultiInputPolicy", env, buffer_size = 1000000,
    #            policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="try_mnist_td3_fltrust_g_black/",
    #            verbose=1, gamma = 1, action_noise = action_noise, learning_rate=linear_schedule(1e-6), learning_starts = 2000, train_freq = (5, "step"), batch_size = 512)

    # model = PPO("MultiInputPolicy", env, n_steps = 50, learning_rate= 1e-8, ent_coef = 0.01,
    #             policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="FRD_MIP_0.95_PPO_optimal_num2_new/",
    #             verbose=1, gamma = 0.99, batch_size = 10)

    model = DQN("MultiInputPolicy", env, learning_rate = 1e-4, learning_starts = 10000, batch_size = 256, exploration_final_eps=0.001,
                gamma = 1, tau = 0.2, train_freq = (5, "step"), policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="RD_MIP_malicious_1_DQN_0.95_dissimilar_lr_1e-4_true/", verbose=1)

    model.learn(total_timesteps=2000000, callback = checkpoint_callback)
