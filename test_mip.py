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
from voting_env_one_round import *

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

import numpy as np
import pandas as pd
import csv

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
from wolframclient.serializers import export


def generate_preference(rng, num_voters, num_issues, p1, p2 = None):
    if p2 == None:
        return rng.choice([False, True], size=(num_voters, num_issues), p =[p1, 1-p1])
    else:
        first_group_num = num_voters//2
        second_group_num = num_voters - first_group_num
        first_group = rng.choice([False, True], size=(first_group_num, num_issues), p =[p1, 1-p1])
        second_group = rng.choice([False, True], size=(second_group_num, num_issues), p =[p2, 1-p2])
        return np.concatenate((first_group, second_group))

def optimal_pander(session, profile, max_diff_for_approval, true_pref, max_issues_pandered):

    session.evaluate('binVec = {(0 | 1) ..};')
    session.evaluate('distanceFunction[a : binVec][b : binVec] := HammingDistance[a, b];')
    session.evaluate('numApprovals[x : {binVec ..}, ys : binVec, k_] := Length[Select[Map[distanceFunction[ys], x], (# <= k) &]];')
    session.evaluate('maximizeApprovals[x : {binVec ..}, k_, penalty_ : (0 &)] := \
      Module[{y}, \
       With[{ys = Array[y, Length[First[x]]]}, \
        Maximize[{numApprovals[x, ys, k] + penalty[ys], \
           Thread[0 <= ys <= 1]}, \
          ys \[Element] Integers] // {First[#], ys /. Last[#]} &]];')
    #version with constraint, via suitably large penalty
    session.evaluate('maximizeApprovals[x_, k_, ref_, j_] := \
      With[{p = -Length[x] - 1}, \
       maximizeApprovals[x, k, \
        If[distanceFunction[ref][#] <= j, 0, p] &]];')\

    result = session.evaluate(f'maximizeApprovals[{profile}, {max_diff_for_approval}, {true_pref}, {max_issues_pandered}]')
    true_result = []
    for i in result[1]:
        true_result.append(i)
    return true_result

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

def test_RD_selfish(session, num_malicious = 1, seed = 0):
    #session = WolframLanguageSession()
    model = DQN.load("RD_MIP_selfish_DQN_0.95_similar_lr_1e-4/rl_model_2000000_steps.zip")
    #model = DQN.load("RD_MIP_0.95_DQN_optimal/rl_model_500000_steps.zip")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 50
    nissues = 9
    num_agent = num_malicious
    #vr = weights
    vr =av
    max_rnd=100

    round = 0
    #pvs = rng.choice([False, True], size=(nvoters, nissues))
    pvs = generate_preference(rng, nvoters, nissues, 0.25)
    public_weight = np.sum(pvs, axis = 0)/nvoters
    voter_majority = []
    voter_minority = []
    for i in public_weight:
        if i > 0.5:
            voter_majority.append(True)
            voter_minority.append(False)
        else:
            voter_majority.append(False)
            voter_minority.append(True)
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - num_malicious)]
    malicious = []
    for i in range(num_agent):
        malicious.append(MaliciousCandidate(nissues, pvs))
    for agent in malicious:
        candidates.append(agent)
    for can in candidates:
        can.private_profile = rng.choice([False, True], size=nissues, p = [0.25,0.75])
        if can not in malicious:
            can.set_public_profile(pvs)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    total_difference = 0
    action_his = []
    for round in range(100):
        pvs_vec = pvs_to_vector(pvs)
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for agent in malicious:
            state = {'pvs': public_weight, 'preference': agent.private_profile,'honesty': [agent.honesty], 'round':round}
            #print(state)
            action,_ = model.predict(state)
            action_his.append(action)
            #action = 0
            malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(agent.private_profile), int(action))
            true_action = []
            for i in malicious_profile:
                if i > 0:
                    true_action.append(True)
                else:
                    true_action.append(False)
            agent.set_public_profile(true_action)
            #agent.set_public_profile(rng.choice([True, False], size=nissues))
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index,_ = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o,_ = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_rd(committee_o, nissues, test = True)
        difference_o = 0
        for i, j in zip(voter_majority, round_result_o):
            if i!=j:
                difference_o+=1

        round_result = get_outcomes_rd(committee, nissues)
        difference = 0
        for i, j in zip(voter_majority, round_result):
            if i!=j:
                difference+=1
        for agent in malicious:
            agree = agreement(agent.private_profile, round_result)
            in_committee = agent in committee
            in_committee_o = agent in committee_o
            #reward = in_committe * 1 + agree * 10
            if in_committee:
                total_chosen +=1
                if not in_committee_o:
                    reward = agree
                else:
                    reward = 0
            else:
                agent.honesty = min(agent.honesty*1.01, 1)
                reward = 0
            #reward = difference - difference_o
            total_reward += reward
        #print('Round', str(round))
        #print("Chosen as committee is ", in_committee)
        #print("honesty is ", str(malicious.honesty))
        # #print("reward is ", reward)
        # print("private", malicious.private_profile)
        # print("public", malicious.public_profile)
        #print(round_result)
        #print(round_result_o)
            if (in_committee) and (not in_committee_o):
                total_chosen_1+=1
        for a,b in zip(round_result, voter_majority):
            if a!=b:
                total_difference+=1
                #print("pandering changes results!")

        #next round
        rng = default_rng(seed = round + seed)
        #pvs = rng.choice([False, True], size=(nvoters, nissues))
        pvs = generate_preference(rng, nvoters, nissues, 0.25)
        for can in candidates:
            can.private_profile = rng.choice([False, True], size=nissues, p = [0.25,0.75])
            if can != malicious:
                can.set_public_profile(pvs)

        voter_majority = []
        voter_minority = []
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for i in public_weight:
            if i > 0.5:
                voter_majority.append(True)
                voter_minority.append(False)
            else:
                voter_majority.append(False)
                voter_minority.append(True)
        #print(public_weight)
        #malicious.private_profile = voter_minority


    #print('total reward is', total_reward)
    #print('total chosen as committee ', total_chosen)
    #print('total chosen by pandering ', total_chosen_1)
    #print('total result changes ', total_difference)
    #session.terminate()

    return total_reward, total_chosen, total_chosen_1, total_difference, action_his


def test_FRD_selfish(session, num_malicious = 1, seed = 0):
    #session = WolframLanguageSession()
    model = DQN.load("FRD_MIP_selfish_DQN_0.95_similar_lr_1e-4/rl_model_2000000_steps.zip")
    #model = DQN.load("RD_MIP_0.95_DQN_optimal/rl_model_500000_steps.zip")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 50
    nissues = 9
    num_agent = num_malicious
    #vr = weights
    vr =av
    max_rnd=100

    round = 0
    #pvs = rng.choice([False, True], size=(nvoters, nissues))
    pvs = generate_preference(rng, nvoters, nissues, 0.25)
    public_weight = np.sum(pvs, axis = 0)/nvoters
    voter_majority = []
    voter_minority = []
    for i in public_weight:
        if i > 0.5:
            voter_majority.append(True)
            voter_minority.append(False)
        else:
            voter_majority.append(False)
            voter_minority.append(True)
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - num_malicious)]
    malicious = []
    for i in range(num_agent):
        malicious.append(MaliciousCandidate(nissues, pvs))
    for agent in malicious:
        candidates.append(agent)
    for can in candidates:
        can.private_profile = rng.choice([False, True], size=nissues, p = [0.25,0.75])
        if can not in malicious:
            can.set_public_profile(pvs)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    total_difference = 0
    action_his = []
    for round in range(100):
        pvs_vec = pvs_to_vector(pvs)
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for agent in malicious:
            state = {'pvs': public_weight, 'preference': agent.private_profile,'honesty': [agent.honesty], 'round':round}
            action,_ = model.predict(state)
            action_his.append(action)
            #action = 0
            malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(agent.private_profile), int(action))
            true_action = []
            for i in malicious_profile:
                if i > 0:
                    true_action.append(True)
                else:
                    true_action.append(False)
            agent.set_public_profile(true_action)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index,_ = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o,_ = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_frd_deterministic(committee_o, nissues, pvs, 0.5, test = True)
        difference_o = 0
        for i, j in zip(voter_majority, round_result_o):
            if i!=j:
                difference_o+=1

        round_result = get_outcomes_frd_deterministic(committee, nissues, pvs, 0.5)
        difference = 0
        for i, j in zip(voter_majority, round_result):
            if i!=j:
                difference+=1
        for agent in malicious:
            agree = agreement(agent.private_profile, round_result)
            in_committee = agent in committee
            in_committee_o = agent in committee_o
            #reward = in_committe * 1 + agree * 10
            if in_committee:
                total_chosen +=1
                if not in_committee_o:
                    reward = agree
                else:
                    reward = 0
            else:
                agent.honesty = min(agent.honesty*1.01, 1)
                reward = 0
            #reward = difference - difference_o
            total_reward += reward
        #print('Round', str(round))
        #print("Chosen as committee is ", in_committee)
        #print("honesty is ", str(malicious.honesty))
        # #print("reward is ", reward)
        # print("private", malicious.private_profile)
        # print("public", malicious.public_profile)
        #print(round_result)
        #print(round_result_o)
            if (in_committee) and (not in_committee_o):
                total_chosen_1+=1
        for a,b in zip(round_result, voter_majority):
            if a!=b:
                total_difference+=1
                #print("pandering changes results!")

        #next round
        rng = default_rng(seed = round + seed)
        #pvs = rng.choice([False, True], size=(nvoters, nissues))
        pvs = generate_preference(rng, nvoters, nissues, 0.25)
        for can in candidates:
            can.private_profile = rng.choice([False, True], size=nissues, p = [0.25,0.75])
            if can != malicious:
                can.set_public_profile(pvs)

        voter_majority = []
        voter_minority = []
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for i in public_weight:
            if i > 0.5:
                voter_majority.append(True)
                voter_minority.append(False)
            else:
                voter_majority.append(False)
                voter_minority.append(True)
        #print(public_weight)
        #malicious.private_profile = voter_minority


    #print('total reward is', total_reward)
    #print('total chosen as committee ', total_chosen)
    #print('total chosen by pandering ', total_chosen_1)
    #print('total result changes ', total_difference)
    #session.terminate()

    return total_reward, total_chosen, total_chosen_1, total_difference, action_his

def test_RD_malicous(session, num_malicious, seed = 0):
    #session = WolframLanguageSession()
    #model = PPO.load("RD_MIP_0.95_PPO_optimal_num3/rl_model_400000_steps.zip")
    #model = DQN.load("RD_MIP_malicious_1_DQN_0.95_dissimilar_lr_1e-4_true/rl_model_2000000_steps.zip")
    model = DQN.load("0.9_result/RD_MIP_0.9_DQN_optimal_num1/rl_model_2000000_steps")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 50
    nissues = 9
    num_agent = num_malicious
    #vr = weights
    vr =av
    max_rnd=1

    round = 0
    #pvs = rng.choice([False, True], size=(nvoters, nissues))
    pvs = generate_preference(rng, nvoters, nissues, 0.5)
    public_weight = np.sum(pvs, axis = 0)/nvoters
    voter_majority = []
    voter_minority = []
    for i in public_weight:
        if i > 0.5:
            voter_majority.append(True)
            voter_minority.append(False)
        else:
            voter_majority.append(False)
            voter_minority.append(True)
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - num_malicious)]
    malicious = []
    for i in range(num_agent):
        malicious.append(MaliciousCandidate(nissues, pvs))
    for agent in malicious:
        candidates.append(agent)
    for can in candidates:
        private_profile = rng.choice([False, True], size=nissues, p = [0.5,0.5])
        if can not in malicious:
            can.set_public_profile(pvs)
    for agent in malicious:
        agent.private_profile = voter_minority

    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    total_difference = 0
    action_his = []
    for round in range(100):
        pvs_vec = pvs_to_vector(pvs)
        public_weight = np.sum(pvs, axis = 0)/nvoters
        honesty = []
        for agent in malicious:
            honesty.append(agent.honesty)
        state = {'pvs': public_weight, 'preference': voter_minority,'honesty': honesty, 'round':round}
        action,_ = model.predict(state)
        action_his.append(action)
        #action = 900
        true_action = reindex(nissues, num_agent, action)
        #action = [0 for _ in range(num_agent)]
        for j in range(num_agent):
            #state = {'pvs': public_weight, 'preference': voter_minority,'honesty': [malicious[j].honesty], 'round':round}
            #action,_ = model.predict(state)
            #malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(action))
            malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(true_action[j]))
            true_action = []
            for i in malicious_profile:
                if i > 0:
                    true_action.append(True)
                else:
                    true_action.append(False)
            malicious[j].set_public_profile(true_action)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index,_ = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o,_ = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_rd(committee_o, nissues, test = True)
        difference_o = 0
        for i, j in zip(voter_majority, round_result_o):
            if i!=j:
                difference_o+=1

        round_result = get_outcomes_rd(committee, nissues)
        difference = 0
        for i, j in zip(voter_majority, round_result):
            if i!=j:
                difference+=1
        for agent in malicious:
            #agree = agreement(agent.private_profile, round_result)
            in_committee = agent in committee
            in_committee_o = agent in committee_o
            #reward = in_committe * 1 + agree * 10
            if in_committee:
                total_chosen +=1
                # if not in_committee_o:
                #     reward = agree * 10
                # else:
                #     reward = agree
            else:
                agent.honesty = min(agent.honesty*1.01, 1)
                #reward = 0
            if (in_committee) and (not in_committee_o):
                total_chosen_1+=1
        reward = difference - difference_o
        total_reward += reward
        for a,b in zip(round_result, voter_majority):
            if a!=b:
                total_difference+=1
                #print("pandering changes results!")

        #next round
        rng = default_rng(seed = round + seed)
        #pvs = rng.choice([False, True], size=(nvoters, nissues))
        pvs = generate_preference(rng, nvoters, nissues, 0.5)
        for can in candidates:
            can.private_profile = rng.choice([False, True], size=nissues, p = [0.5,0.5])
            if can not in malicious:
                can.set_public_profile(pvs)

        voter_majority = []
        voter_minority = []
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for i in public_weight:
            if i > 0.5:
                voter_majority.append(True)
                voter_minority.append(False)
            else:
                voter_majority.append(False)
                voter_minority.append(True)
        #print(public_weight)
        for agent in malicious:
            agent.private_profile = voter_minority
    #print('total reward is', total_reward)
    #print('total chosen as committee ', total_chosen)
    #print('total chosen by pandering ', total_chosen_1)
    #print('total result changes ', total_difference)
    #session.terminate()
    return total_reward, total_chosen, total_chosen_1, total_difference, action_his

def test_FRD_malicous(session, num_malicious, seed = 0):
    #session = WolframLanguageSession()
    #model = DQN.load("FRD_MIP_malicious_1_DQN_0.95_dissimilar_lr_1e-4/rl_model_1000000_steps.zip")
    #model = DQN.load("FRD_MIP_0.95_DQN_optimal_num1/rl_model_500000_steps.zip")
    model = DQN.load("0.9_result/FRD_MIP_0.9_DQN_optimal_num1/rl_model_2000000_steps")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 50
    nissues = 9
    num_agent = num_malicious
    #vr = weights
    vr =av
    max_rnd=100

    round = 0
    #pvs = rng.choice([False, True], size=(nvoters, nissues))
    pvs = generate_preference(rng, nvoters, nissues, 0.5)
    public_weight = np.sum(pvs, axis = 0)/nvoters
    voter_majority = []
    voter_minority = []
    for i in public_weight:
        if i > 0.5:
            voter_majority.append(True)
            voter_minority.append(False)
        else:
            voter_majority.append(False)
            voter_minority.append(True)
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - num_malicious)]
    malicious = []
    for i in range(num_agent):
        malicious.append(MaliciousCandidate(nissues, pvs))
    for agent in malicious:
        candidates.append(agent)
    for can in candidates:
        private_profile = rng.choice([False, True], size=nissues, p = [0.5,0.5])
        if can not in malicious:
            can.set_public_profile(pvs)
    for agent in malicious:
        agent.private_profile = voter_minority
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    total_difference = 0
    action_his = []
    for round in range(100):
        pvs_vec = pvs_to_vector(pvs)
        public_weight = np.sum(pvs, axis = 0)/nvoters
        honesty = []
        for agent in malicious:
            honesty.append(agent.honesty)
        #state = {'pvs': np.asarray(public_weight), 'preference': np.asarray(voter_minority),'honesty': np.asarray(honesty), 'round':round}
        #action,_ = model.predict(state)
        #action = 900
        #true_action = reindex(nissues, num_agent, action)
        #action = [0 for _ in range(num_agent)]
        for j in range(num_agent):
            state = {'pvs': public_weight, 'preference': voter_minority,'honesty': [malicious[j].honesty], 'round':round}
            action,_ = model.predict(state)
            action_his.append(action)
            malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(action))
            #malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(true_action[j]))
            true_action = []
            for i in malicious_profile:
                if i > 0:
                    true_action.append(True)
                else:
                    true_action.append(False)
            malicious[j].set_public_profile(true_action)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index,_ = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o,_ = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_frd_deterministic(committee_o, nissues, pvs, 0.5, test = True)
        difference_o = 0
        for i, j in zip(voter_majority, round_result_o):
            if i!=j:
                difference_o+=1

        round_result = get_outcomes_frd_deterministic(committee, nissues, pvs, 0.5)
        difference = 0
        for i, j in zip(voter_majority, round_result):
            if i!=j:
                difference+=1
        for agent in malicious:
            #agree = agreement(agent.private_profile, round_result)
            in_committee = agent in committee
            in_committee_o = agent in committee_o
            #reward = in_committe * 1 + agree * 10
            if in_committee:
                total_chosen +=1
                # if not in_committee_o:
                #     reward = agree * 10
                # else:
                #     reward = agree
            else:
                agent.honesty = min(agent.honesty*1.01, 1)
                #reward = 0
            if (in_committee) and (not in_committee_o):
                total_chosen_1+=1
        reward = difference - difference_o
        total_reward += reward
        for a,b in zip(round_result, voter_majority):
            if a!=b:
                total_difference+=1
                #print("pandering changes results!")

        #next round
        rng = default_rng(seed = round + seed)
        pvs = generate_preference(rng, nvoters, nissues, 0.5)
        #pvs = rng.choice([False, True], size=(nvoters, nissues))
        for can in candidates:
            can.private_profile = rng.choice([False, True], size=nissues, p = [0.5,0.5])
            if can not in malicious:
                can.set_public_profile(pvs)

        voter_majority = []
        voter_minority = []
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for i in public_weight:
            if i > 0.5:
                voter_majority.append(True)
                voter_minority.append(False)
            else:
                voter_majority.append(False)
                voter_minority.append(True)
        #print(public_weight)
        for agent in malicious:
            agent.private_profile = voter_minority


    #print('total reward is', total_reward)
    #print('total chosen as committee ', total_chosen)
    #print('total chosen by pandering ', total_chosen_1)
    #print('total result changes ', total_difference)
    #session.terminate()

    return total_reward, total_chosen, total_chosen_1, total_difference, action_his


def test_RD_selfish_random(session, num_malicious = 1, seed = 0):
    #session = WolframLanguageSession()
    model = DQN.load("0.95_result/RD_MIP_0.95_DQN_selfish/rl_model_1000000_steps.zip")
    #model = DQN.load("RD_MIP_0.95_DQN_optimal/rl_model_500000_steps.zip")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 50
    nissues = 9
    num_agent = num_malicious
    #vr = weights
    vr =av
    max_rnd=100

    round = 0
    pvs = rng.choice([False, True], size=(nvoters, nissues))
    public_weight = np.sum(pvs, axis = 0)/nvoters
    voter_majority = []
    voter_minority = []
    for i in public_weight:
        if i > 0.5:
            voter_majority.append(True)
            voter_minority.append(False)
        else:
            voter_majority.append(False)
            voter_minority.append(True)
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - num_malicious)]
    malicious = []
    for i in range(num_agent):
        malicious.append(MaliciousCandidate(nissues, pvs))
    for agent in malicious:
        candidates.append(agent)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    total_difference = 0
    action_his = []
    for round in range(100):
        pvs_vec = pvs_to_vector(pvs)
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for agent in malicious:
            state = {'pvs': public_weight, 'preference': agent.private_profile,'honesty': [agent.honesty], 'round':round}
            #action,_ = model.predict(state)
            action = np.random.randint(10)
            action_his.append(action)
            malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(agent.private_profile), int(action))
            true_action = []
            for i in malicious_profile:
                if i > 0:
                    true_action.append(True)
                else:
                    true_action.append(False)
            agent.set_public_profile(rng.choice([True, False], size=nissues))
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index,_ = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o,_ = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_rd(committee_o, nissues, test = True)
        difference_o = 0
        for i, j in zip(voter_majority, round_result_o):
            if i!=j:
                difference_o+=1

        round_result = get_outcomes_rd(committee, nissues)
        difference = 0
        for i, j in zip(voter_majority, round_result):
            if i!=j:
                difference+=1
        for agent in malicious:
            agree = agreement(agent.private_profile, round_result)
            in_committee = agent in committee
            in_committee_o = agent in committee_o
            #reward = in_committe * 1 + agree * 10
            if in_committee:
                total_chosen +=1
                if not in_committee_o:
                    reward = agree
                else:
                    reward = 0
            else:
                agent.honesty = min(agent.honesty*1.01, 1)
                reward = 0
            #reward = difference - difference_o
            total_reward += reward
        #print('Round', str(round))
        #print("Chosen as committee is ", in_committee)
        #print("honesty is ", str(malicious.honesty))
        # #print("reward is ", reward)
        # print("private", malicious.private_profile)
        # print("public", malicious.public_profile)
        #print(round_result)
        #print(round_result_o)
            if (in_committee) and (not in_committee_o):
                total_chosen_1+=1
        for a,b in zip(round_result, voter_majority):
            if a!=b:
                total_difference+=1
                #print("pandering changes results!")

        #next round
        rng = default_rng(seed = round + seed)
        pvs = rng.choice([False, True], size=(nvoters, nissues))
        for can in candidates:
            can.private_profile = rng.choice([True, False], size=nissues)
            if can != malicious:
                can.set_public_profile(pvs)

        voter_majority = []
        voter_minority = []
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for i in public_weight:
            if i > 0.5:
                voter_majority.append(True)
                voter_minority.append(False)
            else:
                voter_majority.append(False)
                voter_minority.append(True)
        #print(public_weight)
        #malicious.private_profile = voter_minority


    #print('total reward is', total_reward)
    #print('total chosen as committee ', total_chosen)
    #print('total chosen by pandering ', total_chosen_1)
    #print('total result changes ', total_difference)
    #session.terminate()

    return total_reward, total_chosen, total_chosen_1, total_difference, action_his

def test_FRD_selfish_random(session, num_malicious = 1, seed = 0):
    #session = WolframLanguageSession()
    model = DQN.load("0.95_result/RD_MIP_0.95_DQN_selfish/rl_model_2000000_steps.zip")
    #model = DQN.load("RD_MIP_0.95_DQN_optimal/rl_model_500000_steps.zip")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 50
    nissues = 9
    num_agent = num_malicious
    #vr = weights
    vr =av
    max_rnd=100

    round = 0
    pvs = rng.choice([False, True], size=(nvoters, nissues))
    public_weight = np.sum(pvs, axis = 0)/nvoters
    voter_majority = []
    voter_minority = []
    for i in public_weight:
        if i > 0.5:
            voter_majority.append(True)
            voter_minority.append(False)
        else:
            voter_majority.append(False)
            voter_minority.append(True)
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - num_malicious)]
    malicious = []
    for i in range(num_agent):
        malicious.append(MaliciousCandidate(nissues, pvs))
    for agent in malicious:
        candidates.append(agent)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    total_difference = 0
    for round in range(100):
        pvs_vec = pvs_to_vector(pvs)
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for agent in malicious:
            state = {'pvs': public_weight, 'preference': agent.private_profile,'honesty': [agent.honesty], 'round':round}
            #action,_ = model.predict(state)
            action = np.random.randint(10)
            malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(agent.private_profile), int(action))
            true_action = []
            for i in malicious_profile:
                if i > 0:
                    true_action.append(True)
                else:
                    true_action.append(False)
            agent.set_public_profile(true_action)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index,_ = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o,_ = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_frd_deterministic(committee_o, nissues, pvs, 0.5, test = True)
        difference_o = 0
        for i, j in zip(voter_majority, round_result_o):
            if i!=j:
                difference_o+=1

        round_result = get_outcomes_frd_deterministic(committee, nissues, pvs, 0.5)
        difference = 0
        for i, j in zip(voter_majority, round_result):
            if i!=j:
                difference+=1
        for agent in malicious:
            agree = agreement(agent.private_profile, round_result)
            in_committee = agent in committee
            in_committee_o = agent in committee_o
            #reward = in_committe * 1 + agree * 10
            if in_committee:
                total_chosen +=1
                if not in_committee_o:
                    reward = agree
                else:
                    reward = 0
            else:
                agent.honesty = min(agent.honesty*1.01, 1)
                reward = 0
            #reward = difference - difference_o
            total_reward += reward
        #print('Round', str(round))
        #print("Chosen as committee is ", in_committee)
        #print("honesty is ", str(malicious.honesty))
        # #print("reward is ", reward)
        # print("private", malicious.private_profile)
        # print("public", malicious.public_profile)
        #print(round_result)
        #print(round_result_o)
            if (in_committee) and (not in_committee_o):
                total_chosen_1+=1
        for a,b in zip(round_result, voter_majority):
            if a!=b:
                total_difference+=1
                #print("pandering changes results!")

        #next round
        rng = default_rng(seed = round + seed)
        pvs = rng.choice([False, True], size=(nvoters, nissues))
        for can in candidates:
            can.private_profile = rng.choice([True, False], size=nissues)
            if can != malicious:
                can.set_public_profile(pvs)

        voter_majority = []
        voter_minority = []
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for i in public_weight:
            if i > 0.5:
                voter_majority.append(True)
                voter_minority.append(False)
            else:
                voter_majority.append(False)
                voter_minority.append(True)
        #print(public_weight)
        #malicious.private_profile = voter_minority


    #print('total reward is', total_reward)
    #print('total chosen as committee ', total_chosen)
    #print('total chosen by pandering ', total_chosen_1)
    #print('total result changes ', total_difference)
    #session.terminate()

    return total_reward, total_chosen, total_chosen_1, total_difference

def test_RD_malicous_random(session, num_malicious, seed = 0):
    #session = WolframLanguageSession()
    model = DQN.load("0.95_result/RD_MIP_0.95_DQN_optimal_num3/rl_model_1000000_steps.zip")
    #model = DQN.load("FRD_MIP_0.95_DQN_optimal_num1/rl_model_500000_steps.zip")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 50
    nissues = 9
    num_agent = num_malicious
    #vr = weights
    vr =av
    max_rnd=100

    round = 0
    pvs = rng.choice([False, True], size=(nvoters, nissues))
    public_weight = np.sum(pvs, axis = 0)/nvoters
    voter_majority = []
    voter_minority = []
    for i in public_weight:
        if i > 0.5:
            voter_majority.append(True)
            voter_minority.append(False)
        else:
            voter_majority.append(False)
            voter_minority.append(True)
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - num_malicious)]
    malicious = []
    for i in range(num_agent):
        malicious.append(MaliciousCandidate(nissues, pvs))
    for agent in malicious:
        agent.private_profile = voter_minority
        candidates.append(agent)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    total_difference = 0
    for round in range(100):
        pvs_vec = pvs_to_vector(pvs)
        public_weight = np.sum(pvs, axis = 0)/nvoters
        honesty = []
        for agent in malicious:
            honesty.append(agent.honesty)
        state = {'pvs': np.asarray(public_weight), 'preference': np.asarray(voter_minority),'honesty': np.asarray(honesty), 'round':round}
        #action,_ = model.predict(state)
        #true_action = reindex(nissues, num_agent, action)
        #action = [0 for _ in range(num_agent)]
        true_action = np.random.randint(10, size=num_agent)
        for j in range(num_agent):
            #state = {'pvs': public_weight, 'preference': voter_minority,'honesty': [malicious[j].honesty], 'round':round}
            #action,_ = model.predict(state)
            #malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(action))
            malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(true_action[j]))
            true_action = []
            for i in malicious_profile:
                if i > 0:
                    true_action.append(True)
                else:
                    true_action.append(False)
            malicious[j].set_public_profile(true_action)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index,_ = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o,_ = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_rd(committee_o, nissues, test = True)
        difference_o = 0
        for i, j in zip(voter_majority, round_result_o):
            if i!=j:
                difference_o+=1

        round_result = get_outcomes_rd(committee, nissues)
        difference = 0
        for i, j in zip(voter_majority, round_result):
            if i!=j:
                difference+=1
        for agent in malicious:
            #agree = agreement(agent.private_profile, round_result)
            in_committee = agent in committee
            in_committee_o = agent in committee_o
            #reward = in_committe * 1 + agree * 10
            if in_committee:
                total_chosen +=1
                # if not in_committee_o:
                #     reward = agree * 10
                # else:
                #     reward = agree
            else:
                agent.honesty = min(agent.honesty*1.01, 1)
                #reward = 0
            if (in_committee) and (not in_committee_o):
                total_chosen_1+=1
        reward = difference - difference_o
        total_reward += reward
        for a,b in zip(round_result, voter_majority):
            if a!=b:
                total_difference+=1
                #print("pandering changes results!")

        #next round
        rng = default_rng(seed = round + seed)
        pvs = rng.choice([False, True], size=(nvoters, nissues))
        for can in candidates:
            can.private_profile = rng.choice([True, False], size=nissues)
            if can != malicious:
                can.set_public_profile(pvs)

        voter_majority = []
        voter_minority = []
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for i in public_weight:
            if i > 0.5:
                voter_majority.append(True)
                voter_minority.append(False)
            else:
                voter_majority.append(False)
                voter_minority.append(True)
        #print(public_weight)
        for agent in malicious:
            agent.private_profile = voter_minority


    #print('total reward is', total_reward)
    #print('total chosen as committee ', total_chosen)
    #print('total chosen by pandering ', total_chosen_1)
    #print('total result changes ', total_difference)
    #session.terminate()

    return total_reward, total_chosen, total_chosen_1, total_difference

def test_FRD_malicous_random(session, num_malicious, seed = 0):
    #session = WolframLanguageSession()
    model = DQN.load("0.95_result/FRD_MIP_0.95_DQN_optimal_num3/rl_model_1000000_steps.zip")
    #model = DQN.load("FRD_MIP_0.95_DQN_optimal_num1/rl_model_500000_steps.zip")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 50
    nissues = 9
    num_agent = num_malicious
    #vr = weights
    vr =av
    max_rnd=100

    round = 0
    pvs = rng.choice([False, True], size=(nvoters, nissues))
    public_weight = np.sum(pvs, axis = 0)/nvoters
    voter_majority = []
    voter_minority = []
    for i in public_weight:
        if i > 0.5:
            voter_majority.append(True)
            voter_minority.append(False)
        else:
            voter_majority.append(False)
            voter_minority.append(True)
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - num_malicious)]
    malicious = []
    for i in range(num_agent):
        malicious.append(MaliciousCandidate(nissues, pvs))
    for agent in malicious:
        agent.private_profile = voter_minority
        candidates.append(agent)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    total_difference = 0
    for round in range(100):
        pvs_vec = pvs_to_vector(pvs)
        public_weight = np.sum(pvs, axis = 0)/nvoters
        honesty = []
        for agent in malicious:
            honesty.append(agent.honesty)
        state = {'pvs': np.asarray(public_weight), 'preference': np.asarray(voter_minority),'honesty': np.asarray(honesty), 'round':round}
        #action,_ = model.predict(state)
        #true_action = reindex(nissues, num_agent, action)
        #action = [0 for _ in range(num_agent)]
        true_action = np.random.randint(10, size=num_agent)
        for j in range(num_agent):
            #state = {'pvs': public_weight, 'preference': voter_minority,'honesty': [malicious[j].honesty], 'round':round}
            #action,_ = model.predict(state)
            #malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(action))
            malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(true_action[j]))
            true_action = []
            for i in malicious_profile:
                if i > 0:
                    true_action.append(True)
                else:
                    true_action.append(False)
            malicious[j].set_public_profile(true_action)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index,_ = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o,_ = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_frd_deterministic(committee_o, nissues, pvs, 0.5, test = True)
        difference_o = 0
        for i, j in zip(voter_majority, round_result_o):
            if i!=j:
                difference_o+=1

        round_result = get_outcomes_frd_deterministic(committee, nissues, pvs, 0.5)
        difference = 0
        for i, j in zip(voter_majority, round_result):
            if i!=j:
                difference+=1
        for agent in malicious:
            #agree = agreement(agent.private_profile, round_result)
            in_committee = agent in committee
            in_committee_o = agent in committee_o
            #reward = in_committe * 1 + agree * 10
            if in_committee:
                total_chosen +=1
                # if not in_committee_o:
                #     reward = agree * 10
                # else:
                #     reward = agree
            else:
                agent.honesty = min(agent.honesty*1.01, 1)
                #reward = 0
            if (in_committee) and (not in_committee_o):
                total_chosen_1+=1
        reward = difference - difference_o
        total_reward += reward
        for a,b in zip(round_result, voter_majority):
            if a!=b:
                total_difference+=1
                #print("pandering changes results!")

        #next round
        rng = default_rng(seed = round + seed)
        pvs = rng.choice([False, True], size=(nvoters, nissues))
        for can in candidates:
            can.private_profile = rng.choice([True, False], size=nissues)
            if can != malicious:
                can.set_public_profile(pvs)

        voter_majority = []
        voter_minority = []
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for i in public_weight:
            if i > 0.5:
                voter_majority.append(True)
                voter_minority.append(False)
            else:
                voter_majority.append(False)
                voter_minority.append(True)
        #print(public_weight)
        for agent in malicious:
            agent.private_profile = voter_minority


    #print('total reward is', total_reward)
    #print('total chosen as committee ', total_chosen)
    #print('total chosen by pandering ', total_chosen_1)
    #print('total result changes ', total_difference)
    #session.terminate()

    return total_reward, total_chosen, total_chosen_1, total_difference

def test_RD_malicous_greedy(session, num_malicious, seed = 0):
    #session = WolframLanguageSession()
    model = DQN.load("0.95_result/FRD_MIP_0.95_DQN_optimal_num3/rl_model_1000000_steps.zip")
    #model = DQN.load("FRD_MIP_0.95_DQN_optimal_num1/rl_model_500000_steps.zip")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 50
    nissues = 9
    num_agent = num_malicious
    #vr = weights
    vr =av
    max_rnd=100

    round = 0
    pvs = rng.choice([False, True], size=(nvoters, nissues))
    public_weight = np.sum(pvs, axis = 0)/nvoters
    voter_majority = []
    voter_minority = []
    for i in public_weight:
        if i > 0.5:
            voter_majority.append(True)
            voter_minority.append(False)
        else:
            voter_majority.append(False)
            voter_minority.append(True)
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - num_malicious)]
    malicious = []
    for i in range(num_agent):
        malicious.append(MaliciousCandidate(nissues, pvs))
    for agent in malicious:
        agent.private_profile = voter_minority
        candidates.append(agent)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    total_difference = 0
    for round in range(100):
        pvs_vec = pvs_to_vector(pvs)
        public_weight = np.sum(pvs, axis = 0)/nvoters
        honesty = []
        for agent in malicious:
            honesty.append(agent.honesty)
        #state = {'pvs': np.asarray(public_weight), 'preference': np.asarray(voter_minority),'honesty': np.asarray(honesty), 'round':round}
        #action,_ = model.predict(state)
        #true_action = reindex(nissues, num_agent, action)
        #action = [0 for _ in range(num_agent)]
        true_action = np.random.randint(10, size=num_agent)
        for j in range(num_agent):
            #state = {'pvs': public_weight, 'preference': voter_minority,'honesty': [malicious[j].honesty], 'round':round}
            #action,_ = model.predict(state)
            #malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(action))
            #malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(true_action[j]))
            true_action = voter_majority
            # for i in malicious_profile:
            #     if i > 0:
            #         true_action.append(True)
            #     else:
            #         true_action.append(False)
            malicious[j].set_public_profile(true_action)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index,_ = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o,_ = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_rd(committee_o, nissues, test = True)
        difference_o = 0
        for i, j in zip(voter_majority, round_result_o):
            if i!=j:
                difference_o+=1

        round_result = get_outcomes_rd(committee, nissues)
        difference = 0
        for i, j in zip(voter_majority, round_result):
            if i!=j:
                difference+=1
        for agent in malicious:
            #agree = agreement(agent.private_profile, round_result)
            in_committee = agent in committee
            in_committee_o = agent in committee_o
            #reward = in_committe * 1 + agree * 10
            if in_committee:
                total_chosen +=1
                # if not in_committee_o:
                #     reward = agree * 10
                # else:
                #     reward = agree
            else:
                agent.honesty = min(agent.honesty*1.01, 1)
                #reward = 0
            if (in_committee) and (not in_committee_o):
                total_chosen_1+=1
        reward = difference - difference_o
        total_reward += reward
        for a,b in zip(round_result, voter_majority):
            if a!=b:
                total_difference+=1
                #print("pandering changes results!")

        #next round
        rng = default_rng(seed = round + seed)
        pvs = rng.choice([False, True], size=(nvoters, nissues))
        for can in candidates:
            can.private_profile = rng.choice([True, False], size=nissues)
            if can != malicious:
                can.set_public_profile(pvs)

        voter_majority = []
        voter_minority = []
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for i in public_weight:
            if i > 0.5:
                voter_majority.append(True)
                voter_minority.append(False)
            else:
                voter_majority.append(False)
                voter_minority.append(True)
        #print(public_weight)
        for agent in malicious:
            agent.private_profile = voter_minority


    #print('total reward is', total_reward)
    #print('total chosen as committee ', total_chosen)
    #print('total chosen by pandering ', total_chosen_1)
    #print('total result changes ', total_difference)
    #session.terminate()

    return total_reward, total_chosen, total_chosen_1, total_difference

def test_RD_malicous_single(session, num_malicious, seed = 0):
    #session = WolframLanguageSession()
    #model = PPO.load("RD_MIP_0.95_PPO_optimal_num3/rl_model_400000_steps.zip")
    model = DQN.load("0.9_result/RD_MIP_0.9_DQN_optimal_num1/rl_model_2000000_steps.zip")
    rng = default_rng(seed = 100)
    random.seed(100)
    np.random.seed(100)
    ncans = 10
    k = 5  # committee size
    nvoters = 50
    nissues = 9
    num_agent = num_malicious
    #vr = weights
    vr =av
    max_rnd=100

    round = 0
    pvs = rng.choice([False, True], size=(nvoters, nissues))
    public_weight = np.sum(pvs, axis = 0)/nvoters
    voter_majority = []
    voter_minority = []
    for i in public_weight:
        if i > 0.5:
            voter_majority.append(True)
            voter_minority.append(False)
        else:
            voter_majority.append(False)
            voter_minority.append(True)
    #print(pvs[0])
    candidates = [HonestCandidate(nissues, pvs) for _ in range(ncans - num_malicious)]
    malicious = []
    for i in range(num_agent):
        malicious.append(MaliciousCandidate(nissues, pvs))
    for agent in malicious:
        agent.private_profile = voter_minority
        candidates.append(agent)
    total_reward = 0
    total_chosen = 0
    total_chosen_1 = 0
    total_difference = 0
    for round in range(100):
        pvs_vec = pvs_to_vector(pvs)
        public_weight = np.sum(pvs, axis = 0)/nvoters
        honesty = []
        for agent in malicious:
            honesty.append(agent.honesty)
        #state = {'pvs': public_weight, 'preference': voter_minority,'honesty': honesty, 'round':round}
        #action,_ = model.predict(state)
        #true_action = reindex(nissues, num_agent, action)
        #action = [0 for _ in range(num_agent)]
        for j in range(num_agent):
            state = {'pvs': public_weight, 'preference': voter_minority,'honesty': [malicious[j].honesty], 'round':round}
            action,_ = model.predict(state)
            malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(action))
            #malicious_profile = optimal_pander(session, list_to_wol(pvs), 4, list_to_wol(voter_minority), int(true_action[j]))
            true_action = []
            for i in malicious_profile:
                if i > 0:
                    true_action.append(True)
                else:
                    true_action.append(False)
            malicious[j].set_public_profile(true_action)
        ppc = []
        for can in candidates:
            ppc.append(can.public_profile)
        pvc = get_preferences_on_candidates_with_malicious(pvs, candidates)
        #print(pvc)
        committee_index,_ = get_committee(pvc, vr, k)
        #print(committee_index)
        committee = []
        for i in committee_index:
            #print(i)
            committee.append(candidates[i-1])
        #print(self.committee)

        pvc_o = get_preferences_on_candidates_without_malicious(pvs, candidates)
        committee_index_o,_ = get_committee(pvc_o, vr, k)
        committee_o = []
        for i in committee_index_o:
            #print(i)
            committee_o.append(candidates[i-1])
        round_result_o = get_outcomes_rd(committee_o, nissues, test = True)
        difference_o = 0
        for i, j in zip(voter_majority, round_result_o):
            if i!=j:
                difference_o+=1

        round_result = get_outcomes_rd(committee, nissues)
        difference = 0
        for i, j in zip(voter_majority, round_result):
            if i!=j:
                difference+=1
        for agent in malicious:
            #agree = agreement(agent.private_profile, round_result)
            in_committee = agent in committee
            in_committee_o = agent in committee_o
            #reward = in_committe * 1 + agree * 10
            if in_committee:
                total_chosen +=1
                # if not in_committee_o:
                #     reward = agree * 10
                # else:
                #     reward = agree
            else:
                agent.honesty = min(agent.honesty*1.01, 1)
                #reward = 0
            if (in_committee) and (not in_committee_o):
                total_chosen_1+=1
        reward = difference - difference_o
        total_reward += reward
        for a,b in zip(round_result, voter_majority):
            if a!=b:
                total_difference+=1
                #print("pandering changes results!")

        #next round
        rng = default_rng(seed = round + seed)
        pvs = rng.choice([False, True], size=(nvoters, nissues))
        for can in candidates:
            can.private_profile = rng.choice([True, False], size=nissues)
            if can != malicious:
                can.set_public_profile(pvs)

        voter_majority = []
        voter_minority = []
        public_weight = np.sum(pvs, axis = 0)/nvoters
        for i in public_weight:
            if i > 0.5:
                voter_majority.append(True)
                voter_minority.append(False)
            else:
                voter_majority.append(False)
                voter_minority.append(True)
        #print(public_weight)
        for agent in malicious:
            agent.private_profile = voter_minority
    #print('total reward is', total_reward)
    #print('total chosen as committee ', total_chosen)
    #print('total chosen by pandering ', total_chosen_1)
    #print('total result changes ', total_difference)
    #session.terminate()
    return total_reward, total_chosen, total_chosen_1, total_difference




seeds = [(i+1)*100 for i in range(10)]


# reward = []
# chosen = []
# chosen_1 = []
# difference = []
# session = WolframLanguageSession()
# for seed in seeds:
#     #print("seed ", seed)
#     a,b,c,d = test_RD_malicous(session, 1, seed)
#     reward.append(a)
#     chosen.append(b)
#     chosen_1.append(c)
#     difference.append(d)
# session.terminate()
#
# print("RD")
# print('total reward is', np.mean(reward), np.var(reward))
# print('total chosen as committee ', np.mean(chosen), np.var(chosen))
# print('total chosen by pandering ', np.mean(chosen_1), np.var(chosen_1))
# print('total difference between voter majority ', np.mean(difference), np.var(difference))
#
# file = open("rd_malicious_1round_1.csv", 'w')
# writer = csv.writer(file)
# writer.writerow(reward)
# writer.writerow(chosen)
# writer.writerow(chosen_1)
# writer.writerow(difference)
# file.close()
#
reward = []
chosen = []
chosen_1 = []
difference = []
action_his = []
session = WolframLanguageSession()
for seed in seeds:
    print("seed ", seed)
    #a,b,c,d,e = test_FRD_selfish(session, 1, seed)
    a,b,c,d,e = test_FRD_malicous(session, 1, seed)
    reward.append(a)
    chosen.append(b)
    chosen_1.append(c)
    difference.append(d)
    action_his.append(e)
session.terminate()

print("FRD")
print('total reward is', np.mean(reward), np.var(reward))
print('total chosen as committee ', np.mean(chosen), np.var(chosen))
print('total chosen by pandering ', np.mean(chosen_1), np.var(chosen_1))
print('total difference between voter majority ', np.mean(difference), np.var(difference))
print('action history ', np.mean(action_his), np.var(action_his))

file = open("frd_malicious_1_0.9_original.csv", 'w')
writer = csv.writer(file)
writer.writerow(reward)
writer.writerow(chosen)
writer.writerow(chosen_1)
writer.writerow(difference)
writer.writerow(action_his)
file.close()
