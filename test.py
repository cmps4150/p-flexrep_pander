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
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from voting_env import *

def test_RD():
    model = SAC.load("RD_0.9pen_1.003regen_borda_1/rl_model_60000_steps.zip")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 100
    nissues = 30
    #vr = weights
    vr =borda
    max_rnd=100

    round = 0
    pvs = rng.choice([False, True], size=(nvoters, nissues))
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - 1)]
    malicious = MaliciousCandidate(nissues, pvs)
    candidates.append(malicious)
    diff = diff_public_attacker(nissues, pvs, malicious.private_profile, nvoters)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    for round in range(100):
        action,_ = model.predict([len(diff), malicious.honesty], deterministic = True)

        num_change = int(action*len(diff))
        #print(self.diff)
        change_list = [diff[i][0] for i in range(num_change)]
        malicious.set_public_profile(change_list)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)
        round_result = get_outcomes_rd(committee, nissues)
        if num_change > 0:
            pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
            committee_index_o = get_committee(pvc_o, vr, k)
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
            if num_change >0:
                reward = in_committe * (1 + agree * 10)
            else:
                reward = agree * 3
        else:
            malicious.honesty = min(malicious.honesty*1.01, 1)
            reward = agree
        total_reward += reward
        print('Round', str(round))
        print("Chosen as committee is ", in_committe)
        print("action is ", action," ", num_change)
        print("honesty is ", str(malicious.honesty))
        #print("reward is ", reward)
        if num_change > 0:
            in_committe_o = malicious in committee_o
            print("Chosen as committee without pandering is ", in_committe_o)
            if (in_committe) and (not in_committe_o):
                total_chosen_1+=1
            for a,b in zip(round_result, round_result_o):
                if a!=b:
                    print("pandering changes results!")
            print(committee_index)
            print(committee_index_o)
        #next round
        pvs = rng.choice([False, True], size=(nvoters, nissues))
        for can in candidates:
            can.private_profile = rng.choice([True, False], size=nissues)
            if can != malicious:
                can.set_public_profile(pvs)
        diff = diff_public_attacker(nissues, pvs, malicious.private_profile, nvoters)


    print('total reward is', total_reward)
    print('total chosen as committee ', total_chosen)
    print('total chosen by pandering ', total_chosen_1)

def test_FRD():
    model = SAC.load("FRD_0.9pen_1.003regen_borda_1/rl_model_40000_steps.zip")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 100
    nissues = 30
    #vr = weights
    vr = borda
    max_rnd=100

    round = 0
    pvs = rng.choice([False, True], size=(nvoters, nissues))
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - 1)]
    malicious = MaliciousCandidate(nissues, pvs)
    candidates.append(malicious)
    diff = diff_public_attacker(nissues, pvs, malicious.private_profile, nvoters)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    for round in range(100):
        action,_ = model.predict([len(diff), malicious.honesty], deterministic = True)

        num_change = int(action*len(diff))
        #print(self.diff)
        change_list = [diff[i][0] for i in range(num_change)]
        malicious.set_public_profile(change_list)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        committee_index = get_committee(pvc, vr, k)
        #print(self.committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)
        round_result = get_outcomes_frd(committee, nissues, pvs, 0.5)
        if num_change > 0:
            pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
            committee_index_o = get_committee(pvc_o, vr, k)
            committee_o = []
            for i in committee_index_o:
                #print(i)
                committee_o.append(candidates[i-1])
            round_result_o = get_outcomes_frd(committee_o, nissues, pvs, 0.5, test = True)

        agree = agreement(malicious.private_profile, round_result)
        in_committe = malicious in committee
        #reward = in_committe * 1 + agree * 10
        if in_committe:
            total_chosen +=1
            if num_change >0:
                reward = in_committe * (1 + agree * 10)
            else:
                reward = agree * 3
        else:
            malicious.honesty = min(malicious.honesty*1.01, 1)
            reward = agree
        total_reward += reward
        print('Round', str(round))
        print("Chosen as committee is ", in_committe)
        print("action is ", action," ", num_change)
        print("honesty is ", str(malicious.honesty))
        #print("reward is ", reward)
        if num_change > 0:
            in_committe_o = malicious in committee_o
            print("Chosen as committee without pandering is ", in_committe_o)
            if (in_committe) and (not in_committe_o):
                total_chosen_1+=1
            for a,b in zip(round_result, round_result_o):
                if a!=b:
                    print("pandering changes results!")
            print(committee_index)
            print(committee_index_o)
        #next round
        pvs = rng.choice([False, True], size=(nvoters, nissues))
        for can in candidates:
            can.private_profile = rng.choice([True, False], size=nissues)
            if can != malicious:
                can.set_public_profile(pvs)
        diff = diff_public_attacker(nissues, pvs, malicious.private_profile, nvoters)


    print('total reward is', total_reward)
    print('total chosen as committee ', total_chosen)
    print('total chosen by pandering ', total_chosen_1)

def test_RD_multi():
    model = SAC.load("RD_multi_0.9pen_1.003regen_weights_2/rl_model_180000_steps.zip")
    rng = default_rng(seed = 12)
    random.seed(12)
    np.random.seed(12)
    ncans = 20
    k = 5  # committee size
    nvoters = 100
    nissues = 30
    nattackers = 2
    vr = weights
    #vr =borda
    max_rnd=100

    round = 0
    pvs = rng.choice([False, True], size=(nvoters, nissues))
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - nattackers)]
    private = rng.choice([True, False], size=nissues)
    malicious = [MaliciousCandidate(nissues, pvs, private = private) for _ in range(nattackers)]
    for str_can in malicious:
        candidates.append(str_can)
    diff = diff_public_attacker(nissues, pvs, malicious[0].private_profile, nvoters)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    for round in range(100):
        state = [len(diff)]
        for str_can in malicious:
            state.append(str_can.honesty)
        action,_ = model.predict(state, deterministic = True)
        #num_change = int(action*len(diff))
        num_change = []
        for act in action:
            num_change.append(int(act*len(diff)))
        #print(self.diff)
        change_list = []
        for i in num_change:
            change_list.append([diff[j][0] for j in range(i)])
        #change_list = [diff[i][0] for i in range(num_change)]
        for str_can, change in zip(malicious, change_list):
            str_can.set_public_profile(change)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)
        round_result = get_outcomes_rd(committee, nissues)
        committee_o = []

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_rd(committee_o, nissues, test = True)

        agree = agreement(private, round_result)
        in_committe = False
        for str_can in malicious:
            if str_can in committee:
                in_committe = True
                break

        in_committe_o = False
        for str_can in malicious:
            if str_can in committee_o:
                in_committe_o = True
                break
        #in_committe = malicious in committee
        #reward = in_committe * 1 + agree * 10
        if in_committe:
            total_chosen +=1
            if in_committe and (not in_committe_o):
                reward = in_committe * (1 + agree * 10)
            else:
                reward = agree * 3
        else:
            #malicious.honesty = min(malicious.honesty*1.01, 1)
            reward = agree
        for str_can in malicious:
            if str_can not in committee:
                str_can.honesty = min(str_can.honesty*1.01, 1)

        total_reward += reward
        print('Round', str(round))
        print("Chosen as committee is ", in_committe)
        print("action is ", action," ", num_change)
        print("honesty is ", [str_can.honesty for str_can in malicious])
        #print("reward is ", reward)
        if sum(num_change) > 0:
            #in_committe_o = malicious in committee_o

            print("Chosen as committee without pandering is ", in_committe_o)
            if (in_committe) and (not in_committe_o):
                total_chosen_1+=1
            for a,b in zip(round_result, round_result_o):
                if a!=b:
                    print("pandering changes results!")
            print(committee_index)
            print(committee_index_o)
        #next round
        pvs = rng.choice([False, True], size=(nvoters, nissues))
        private = rng.choice([True, False], size=nissues)
        for can in candidates:
            if can not in malicious:
                can.private_profile = rng.choice([True, False], size=nissues)
                can.set_public_profile(pvs)
            else:
                can.private_profile = private
        diff = diff_public_attacker(nissues, pvs, malicious[0].private_profile, nvoters)

    print('total reward is', total_reward)
    print('total chosen as committee ', total_chosen)
    print('total chosen by pandering ', total_chosen_1)

def test_FRD_multi():
    model = SAC.load("FRD_multi_0.9pen_1.003regen_weights/rl_model_100000_steps.zip")
    rng = default_rng(seed = 12)
    random.seed(12)
    np.random.seed(12)
    ncans = 20
    k = 5  # committee size
    nvoters = 100
    nissues = 30
    nattackers = 2
    vr = weights
    #vr =borda
    max_rnd=100

    round = 0
    pvs = rng.choice([False, True], size=(nvoters, nissues))
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - nattackers)]
    private = rng.choice([True, False], size=nissues)
    malicious = [MaliciousCandidate(nissues, pvs, private = private) for _ in range(nattackers)]
    for str_can in malicious:
        candidates.append(str_can)
    diff = diff_public_attacker(nissues, pvs, malicious[0].private_profile, nvoters)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    for round in range(100):
        state = [len(diff)]
        for str_can in malicious:
            state.append(str_can.honesty)
        action,_ = model.predict(state, deterministic = True)
        #num_change = int(action*len(diff))
        num_change = []
        for act in action:
            num_change.append(int(act*len(diff)))
        #print(self.diff)
        change_list = []
        for i in num_change:
            change_list.append([diff[j][0] for j in range(i)])
        #change_list = [diff[i][0] for i in range(num_change)]
        for str_can, change in zip(malicious, change_list):
            str_can.set_public_profile(change)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)
        round_result = get_outcomes_frd(committee, nissues, pvs, 0.5)
        committee_o = []

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_frd(committee_o, nissues, pvs, 0.5, test = True)

        agree = agreement(private, round_result)
        in_committe = False
        for str_can in malicious:
            if str_can in committee:
                in_committe = True
                break

        in_committe_o = False
        for str_can in malicious:
            if str_can in committee_o:
                in_committe_o = True
                break
        #in_committe = malicious in committee
        #reward = in_committe * 1 + agree * 10
        if in_committe:
            total_chosen +=1
            if in_committe and (not in_committe_o):
                reward = in_committe * (1 + agree * 10)
            else:
                reward = agree * 3
        else:
            #malicious.honesty = min(malicious.honesty*1.01, 1)
            reward = agree
        for str_can in malicious:
            if str_can not in committee:
                str_can.honesty = min(str_can.honesty*1.01, 1)

        total_reward += reward
        print('Round', str(round))
        print("Chosen as committee is ", in_committe)
        print("action is ", action," ", num_change)
        print("honesty is ", [str_can.honesty for str_can in malicious])
        #print("reward is ", reward)
        if sum(num_change) > 0:
            #in_committe_o = malicious in committee_o

            print("Chosen as committee without pandering is ", in_committe_o)
            if (in_committe) and (not in_committe_o):
                total_chosen_1+=1
            for a,b in zip(round_result, round_result_o):
                if a!=b:
                    print("pandering changes results!")
            print(committee_index)
            print(committee_index_o)
        #next round
        pvs = rng.choice([False, True], size=(nvoters, nissues))
        private = rng.choice([True, False], size=nissues)
        for can in candidates:
            if can not in malicious:
                can.private_profile = rng.choice([True, False], size=nissues)
                can.set_public_profile(pvs)
            else:
                can.private_profile = private
        diff = diff_public_attacker(nissues, pvs, malicious[0].private_profile, nvoters)

    print('total reward is', total_reward)
    print('total chosen as committee ', total_chosen)
    print('total chosen by pandering ', total_chosen_1)

print("results on RD")
test_RD_multi()
print("results on FRD")
test_FRD_multi()
