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
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from train_voting import *

def test_RD():
    model = PPO.load("RD_oneround_votes_allobs_1/rl_model_10000000_steps.zip")
    rng = default_rng(seed = 10000)
    random.seed(10000)
    np.random.seed(10000)
    ncans = 10
    k = 5  # committee size
    nvoters = 19
    nissues = 9
    #vr = weights
    vr =av
    #max_rnd=5
    rl_optimal = 0
    greedy_optimal = 0
    round = 0
    pvs = rng.choice([False, True], size=(nvoters, nissues))
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - 1)]
    malicious = MaliciousCandidate(nissues, pvs)
    candidates.append(malicious)
    #diff = diff_public_attacker(nissues, pvs, malicious.private_profile, nvoters)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0

    for round in range(100):
        public_weight = np.sum(pvs, axis = 0)
        pvs_vec = pvs_to_vector(pvs)
        action,_ = model.predict(pvs_vec, deterministic = True)
        rl_action = []
        for i in action:
            if i > 0:
                rl_action.append(True)
            else:
                rl_action.append(False)
        #true_action =  rng.choice([False, True], size=nissues)
        #true_action = malicious.private_profile

        #Greedy
        greedy_action = []
        for i in public_weight:
            if i > nvoters/2:
                greedy_action.append(True)
            elif i < nvoters/2:
                greedy_action.append(False)
            elif i == nvoters/2:
                #print(rng.choice([False, True]))
                greedy_action.append(rng.choice([False, True]))

        #optimal
        all_possible_action = []
        for i in range(2**nissues):
            o_bin = bin(i)[2:]
            real_bin = o_bin.rjust(nissues, '0')
            #print(real_bin)
            action = []
            for j in real_bin:
                if j=="0":
                    action.append(True)
                else:
                    action.append(False)
            #print(action)
            all_possible_action.append(action)
        max_value = -100
        true_action = None
        equal_action = []
        for action in all_possible_action:
            malicious.set_public_profile(action)
            ppc = []
            for can in candidates:
                ppc.append(can.public_profile)
            pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
            #print(pvc)
            committee_index, malicious_score = get_committee(pvc, vr, k)
            #print(committee_index)
            committee = []
            for i in committee_index:
                #print(i)
                committee.append(candidates[i-1])
            #print(self.committee)
            round_result = get_outcomes_rd(committee, nissues, test = True)
            agree = agreement(malicious.private_profile, round_result)
            in_committe = malicious in committee
            #reward = in_committe * 1 + agree * 10
            # if in_committe:
            #     #total_chosen +=1
            #     # for a,b in zip(true_action, malicious.private_profile):
            #     #     if a != b:
            #     #         total_chosen_1 += 1
            #     #         break
            #     reward = agree
            # else:
            #     reward = 0
            reward = malicious_score
            if reward > max_value:
                true_action = action
                equal_action = []
                equal_action.append(action)
                max_value = reward
            elif reward == max_value:
                equal_action.append(action)





        #num_change = int(action*len(diff))
        #print(self.diff)
        #change_list = [diff[i][0] for i in range(num_change)]
        malicious.set_public_profile(true_action)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index, malicious_score = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)
        round_result = get_outcomes_rd(committee, nissues, test = True)
        # if num_change > 0:
        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o, _ = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_rd(committee_o, nissues, test = True)

        agree = agreement(malicious.private_profile, round_result)
        in_committe = malicious in committee
        #reward = in_committe * 1 + agree * 10
        if in_committe:
            total_chosen +=1
            # for a,b in zip(true_action, malicious.private_profile):
            #     if a != b:
            #         total_chosen_1 += 1
            #         break
            reward = agree
        else:
            reward = 0
        reward = malicious_score

        total_reward += reward
        print('Round', str(round))
        print("Chosen as committee is ", in_committe)
        #print("action is ", action," ", num_change)
        #print("honesty is ", str(malicious.honesty))
        print("reward is ", reward)
        in_committe_o = malicious in committee_o
        print("Chosen as committee without pandering is ", in_committe_o)
        if (in_committe) and (not in_committe_o):
            total_chosen_1+=1
        for a,b in zip(round_result, round_result_o):
            if a!=b:
                print("pandering changes results!")
        #print(committee_index)
        #print(committee_index_o)
        # print("rl_action", rl_action in equal_action)
        # print("greedy_action", greedy_action in equal_action)
        # print("optimal_action", len(equal_action))
        if rl_action in equal_action:
            rl_optimal += 1
        if greedy_action in equal_action:
            greedy_optimal += 1


        #next round
        pvs = rng.choice([False, True], size=(nvoters, nissues))
        for can in candidates:
            can.private_profile = rng.choice([True, False], size=nissues)
            if can != malicious:
                can.set_public_profile(pvs)
        #diff = diff_public_attacker(nissues, pvs, malicious.private_profile, nvoters)

    print('rl_optimal', rl_optimal)
    print('greedy_optimal', greedy_optimal)
    print('total reward is', total_reward)
    print('total chosen as committee ', total_chosen)
    print('total chosen by pandering ', total_chosen_1)

test_RD()
