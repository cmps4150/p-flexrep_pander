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
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback


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
        # self.public_profile = copy.deepcopy(self.private_profile)
        # for i in changelist:
        #     if self.public_profile[i]:
        #         self.public_profile[i] = False
        #     else:
        #         self.public_profile[i] = True
            #self.public_profile[i] = -self.public_profile[i]
        self.public_profile = action

class HonestCandidate(Candidate):
    def set_public_profile(self, pvs):
        self.public_profile = self.private_profile



class RD_voting(gym.Env):

    def __init__(self):
        #initilize
        self.rng = default_rng()
        self.ncans = 10
        self.k = 5  # committee size
        self.nvoters = 100
        self.nissues = 30
        #self.vr = weights
        self.vr = borda
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

    def seed(self, seed=None):
        pass

    def step(self, action):
        #print(self.candidates[0].honesty)
        num_change = int(action*len(self.diff))
        #print(self.diff)
        change_list = [self.diff[i][0] for i in range(num_change)]
        self.malicious.set_public_profile(change_list)
        self.ppc = []
        for can in self.candidates:
            self.ppc.append(can.public_profile)
        self.pvc = get_preferences_on_candidates_with_malicious(self.pvs, self.candidates)
        self.committee_index = get_committee(self.pvc, self.vr, self.k)
        #print(self.committee_index)
        self.committee = []
        for i in self.committee_index:
            #print(i)
            self.committee.append(self.candidates[i-1])
        #print(self.committee)
        self.round_result = get_outcomes_rd(self.committee, self.nissues)
        agree = agreement(self.malicious.private_profile, self.round_result)
        in_committe = self.malicious in self.committee
        #reward = in_committe * 1 + agree * 10
        if in_committe:
            if num_change>0:
                reward = 1 + agree * 10
                self.chosen_1 += 1
            else:
                reward = agree * 3
            self.chosen+=1
        else:
            self.malicious.honesty = min(self.malicious.honesty*1.01, 1)
            reward = agree
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
            print("Chosen as committee is ", str(self.chosen))
            print("Chosen as committee by pandering is ", self.chosen_1)
            print("Final honesty is ", str(self.malicious.honesty))
        else:
            done = False


        return [len(self.diff), self.malicious.honesty], reward, done, {}


    def reset(self):
        self.round = 0
        self.chosen = 0
        self.chosen_1 = 0
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.candidates = [HonestCandidate(self.nissues, self.pvs) for _ in range(self.ncans - 1)]
        self.malicious = MaliciousCandidate(self.nissues, self.pvs)
        self.candidates.append(self.malicious)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        #print(self.diff)

        return [len(self.diff), self.malicious.honesty]


class FRD_voting(gym.Env):

    def __init__(self):
        #initilize
        self.rng = default_rng()
        self.ncans = 10
        self.k = 5  # committee size
        self.nvoters = 100
        self.nissues = 30
        #self.vr = weights
        self.vr = borda
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

    def seed(self, seed=None):
        pass

    def step(self, action):
        #print(self.candidates[0].honesty)
        num_change = int(action*len(self.diff))
        #print(self.diff)
        change_list = [self.diff[i][0] for i in range(num_change)]
        self.malicious.set_public_profile(change_list)
        self.ppc = []
        for can in self.candidates:
            self.ppc.append(can.public_profile)
        self.pvc = get_preferences_on_candidates_with_malicious(self.pvs, self.candidates)
        self.committee_index = get_committee(self.pvc, self.vr, self.k)
        #print(self.committee_index)
        self.committee = []
        for i in self.committee_index:
            #print(i)
            self.committee.append(self.candidates[i-1])
        #print(self.committee)
        self.round_result = get_outcomes_frd(self.committee, self.nissues, self.pvs, 0.5)
        agree = agreement(self.malicious.private_profile, self.round_result)
        in_committe = self.malicious in self.committee
        #reward = in_committe * 1 + agree * 10
        if in_committe:
            if num_change>0:
                reward = 1 + agree * 10
                self.chosen_1 += 1
            else:
                reward = agree * 3
            self.chosen+=1
        else:
            self.malicious.honesty = min(self.malicious.honesty*1.01, 1)
            reward = agree
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
            print("Chosen as committee is ", str(self.chosen))
            print("Chosen as committee by pandering is ", self.chosen_1)
            print("Final honesty is ", str(self.malicious.honesty))
        else:
            done = False


        return [len(self.diff), self.malicious.honesty], reward, done, {}


    def reset(self):
        self.round = 0
        self.chosen = 0
        self.chosen_1 = 0
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.candidates = [HonestCandidate(self.nissues, self.pvs) for _ in range(self.ncans - 1)]
        self.malicious = MaliciousCandidate(self.nissues, self.pvs)
        self.candidates.append(self.malicious)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        #print(self.diff)

        return [len(self.diff), self.malicious.honesty]

class RD_voting_multi(gym.Env):

    def __init__(self):
        #initilize
        self.rng = default_rng()
        self.ncans = 20
        self.num_strategic = 2
        self.k = 5  # committee size
        self.nvoters = 100
        self.nissues = 30
        self.vr = weights
        #self.vr = borda
        self.max_rnd=100
        high = 1
        low = 0
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(self.num_strategic,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(1+self.num_strategic,), dtype = np.float32)
        #self.seed()

    def seed(self, seed=None):
        pass

    def step(self, action):
        #print(self.candidates[0].honesty)
        num_change = []
        for act in action:
            num_change.append(int(act*len(self.diff)))
        #print(num_change)
        #num_change = int(action*len(self.diff))
        #print(self.diff)
        #change_list = [self.diff[i][0] for i in range(num_change)]
        change_list = []
        for i in num_change:
            change_list.append([self.diff[j][0] for j in range(i)])
        #print(change_list)
        for i in range(len(self.malicious)):
            self.malicious[i].set_public_profile(change_list[i])
        #self.malicious.set_public_profile(change_list)
        self.ppc = []
        for can in self.candidates:
            self.ppc.append(can.public_profile)
        self.pvc = get_preferences_on_candidates_with_malicious(self.pvs, self.candidates)
        self.committee_index = get_committee(self.pvc, self.vr, self.k)
        #print(self.committee_index)
        self.committee = []
        for i in self.committee_index:
            #print(i)
            self.committee.append(self.candidates[i-1])
        #print(self.committee)
        #self.round_result = get_outcomes_frd(self.committee, self.nissues)
        self.round_result = get_outcomes_rd(self.committee, self.nissues)
        agree = agreement(self.malicious[0].private_profile, self.round_result)
        in_committe = False
        for str_can in self.malicious:
            if str_can in self.committee:
                in_committe = True
                break

        self.pvc_o = get_preferences_on_candidates_without_malicious(self.pvs, self.candidates)
        self.committee_index_o = get_committee(self.pvc_o, self.vr, self.k)
        self.committee_o = []
        for i in self.committee_index_o:
            #print(i)
            self.committee_o.append(self.candidates[i-1])
        self.round_result_o = get_outcomes_rd(self.committee_o, self.nissues, test = True)
        #in_committe = self.malicious in self.committee
        #reward = in_committe * 1 + agree * 10
        in_committe_o = False
        for str_can in self.malicious:
            if str_can in self.committee_o:
                in_committe_o = True
                break
        if in_committe:
            if sum(num_change)>0 and (not in_committe_o):
                reward = 1 + agree * 10
                self.chosen_1 += 1
            else:
                reward = agree * 3
            self.chosen+=1
        else:
            #self.malicious.honesty = min(self.malicious.honesty*1.01, 1)
            reward = agree

        for str_can in self.malicious:
            if not (str_can in self.committee):
                str_can.honesty = min(str_can.honesty*1.01, 1)

        self.round += 1
        #next round
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        private = self.rng.choice([True, False], size=self.nissues)
        for can in self.candidates:
            if can not in self.malicious:
                can.private_profile = self.rng.choice([True, False], size=self.nissues)
                can.set_public_profile(self.pvs)
            else:
                can.private_profile = private
        self.diff = diff_public_attacker(self.nissues, self.pvs, private, self.nvoters)

        if self.round >= self.max_rnd:
            done = True
            print("Chosen as committee is ", str(self.chosen))
            print("Chosen as committee by pandering is ", self.chosen_1)
            print("Final honesty is ", ([str_can.honesty for str_can in self.malicious]))
        else:
            done = False
        state = [len(self.diff)]
        for str_can in self.malicious:
            state.append(str_can.honesty)

        return state, reward, done, {}


    def reset(self):
        self.round = 0
        self.chosen = 0
        self.chosen_1 = 0
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.candidates = [HonestCandidate(self.nissues, self.pvs) for _ in range(self.ncans - self.num_strategic)]
        private = self.rng.choice([True, False], size=self.nissues)
        self.malicious = [MaliciousCandidate(self.nissues, self.pvs, private = private) for _ in range(self.num_strategic)]
        for str_can in self.malicious:
            self.candidates.append(str_can)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious[0].private_profile, self.nvoters)
        #print(self.diff)

        state = [len(self.diff)]
        for str_can in self.malicious:
            state.append(str_can.honesty)
        return state

class FRD_voting_multi(gym.Env):

    def __init__(self):
        #initilize
        self.rng = default_rng()
        self.ncans = 20
        self.num_strategic = 2
        self.k = 5  # committee size
        self.nvoters = 100
        self.nissues = 30
        self.vr = weights
        #self.vr = borda
        self.max_rnd=100
        high = 1
        low = 0
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(self.num_strategic,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape=(1+self.num_strategic,), dtype = np.float32)
        #self.seed()

    def seed(self, seed=None):
        pass

    def step(self, action):
        #print(self.candidates[0].honesty)
        num_change = []
        for act in action:
            num_change.append(int(act*len(self.diff)))
        #print(num_change)
        #num_change = int(action*len(self.diff))
        #print(self.diff)
        #change_list = [self.diff[i][0] for i in range(num_change)]
        change_list = []
        for i in num_change:
            change_list.append([self.diff[j][0] for j in range(i)])
        #print(change_list)
        for i in range(len(self.malicious)):
            self.malicious[i].set_public_profile(change_list[i])
        #self.malicious.set_public_profile(change_list)
        self.ppc = []
        for can in self.candidates:
            self.ppc.append(can.public_profile)
        self.pvc = get_preferences_on_candidates_with_malicious(self.pvs, self.candidates)
        self.committee_index = get_committee(self.pvc, self.vr, self.k)
        #print(self.committee_index)
        self.committee = []
        for i in self.committee_index:
            #print(i)
            self.committee.append(self.candidates[i-1])
        #print(self.committee)
        #self.round_result = get_outcomes_frd(self.committee, self.nissues)
        self.round_result = get_outcomes_frd(self.committee, self.nissues, self.pvs, 0.5)
        agree = agreement(self.malicious[0].private_profile, self.round_result)
        in_committe = False
        for str_can in self.malicious:
            if str_can in self.committee:
                in_committe = True
                break

        self.pvc_o = get_preferences_on_candidates_without_malicious(self.pvs, self.candidates)
        self.committee_index_o = get_committee(self.pvc_o, self.vr, self.k)
        self.committee_o = []
        for i in self.committee_index_o:
            #print(i)
            self.committee_o.append(self.candidates[i-1])
        self.round_result_o = get_outcomes_frd(self.committee_o, self.nissues, self.pvs, 0.5, test = True)
        #in_committe = self.malicious in self.committee
        #reward = in_committe * 1 + agree * 10
        in_committe_o = False
        for str_can in self.malicious:
            if str_can in self.committee_o:
                in_committe_o = True
                break
        if in_committe:
            if sum(num_change)>0 and (not in_committe_o):
                reward = 1 + agree * 10
                self.chosen_1 += 1
            else:
                reward = agree * 3
            self.chosen+=1
        else:
            #self.malicious.honesty = min(self.malicious.honesty*1.01, 1)
            reward = agree

        for str_can in self.malicious:
            if not (str_can in self.committee):
                str_can.honesty = min(str_can.honesty*1.01, 1)

        self.round += 1
        #next round
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        private = self.rng.choice([True, False], size=self.nissues)
        for can in self.candidates:
            if can not in self.malicious:
                can.private_profile = self.rng.choice([True, False], size=self.nissues)
                can.set_public_profile(self.pvs)
            else:
                can.private_profile = private
        self.diff = diff_public_attacker(self.nissues, self.pvs, private, self.nvoters)

        if self.round >= self.max_rnd:
            done = True
            print("Chosen as committee is ", str(self.chosen))
            print("Chosen as committee by pandering is ", self.chosen_1)
            print("Final honesty is ", ([str_can.honesty for str_can in self.malicious]))
        else:
            done = False
        state = [len(self.diff)]
        for str_can in self.malicious:
            state.append(str_can.honesty)

        return state, reward, done, {}


    def reset(self):
        self.round = 0
        self.chosen = 0
        self.chosen_1 = 0
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.candidates = [HonestCandidate(self.nissues, self.pvs) for _ in range(self.ncans - self.num_strategic)]
        private = self.rng.choice([True, False], size=self.nissues)
        self.malicious = [MaliciousCandidate(self.nissues, self.pvs, private = private) for _ in range(self.num_strategic)]
        for str_can in self.malicious:
            self.candidates.append(str_can)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious[0].private_profile, self.nvoters)
        #print(self.diff)

        state = [len(self.diff)]
        for str_can in self.malicious:
            state.append(str_can.honesty)
        return state


class RD_voting_MIP(gym.Env):

    def __init__(self):
        #initilize
        self.rng = default_rng()
        self.ncans = 10
        self.k = 5  # committee size
        self.nvoters = 100
        self.nissues = 30
        #self.vr = weights
        self.vr = approval
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

    def seed(self, seed=None):
        pass

    def step(self, action):
        #print(self.candidates[0].honesty)
        num_change = int(action*len(self.diff))
        #print(self.diff)
        change_list = [self.diff[i][0] for i in range(num_change)]
        self.malicious.set_public_profile(change_list)
        self.ppc = []
        for can in self.candidates:
            self.ppc.append(can.public_profile)
        self.pvc = get_preferences_on_candidates_with_malicious(self.pvs, self.candidates)
        self.committee_index = get_committee(self.pvc, self.vr, self.k)
        #print(self.committee_index)
        self.committee = []
        for i in self.committee_index:
            #print(i)
            self.committee.append(self.candidates[i-1])
        #print(self.committee)
        self.round_result = get_outcomes_rd(self.committee, self.nissues)
        agree = agreement(self.malicious.private_profile, self.round_result)
        in_committe = self.malicious in self.committee
        #reward = in_committe * 1 + agree * 10
        if in_committe:
            if num_change>0:
                reward = 1 + agree * 10
                self.chosen_1 += 1
            else:
                reward = agree * 3
            self.chosen+=1
        else:
            self.malicious.honesty = min(self.malicious.honesty*1.01, 1)
            reward = agree
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
            print("Chosen as committee is ", str(self.chosen))
            print("Chosen as committee by pandering is ", self.chosen_1)
            print("Final honesty is ", str(self.malicious.honesty))
        else:
            done = False


        return [len(self.diff), self.malicious.honesty], reward, done, {}


    def reset(self):
        self.round = 0
        self.chosen = 0
        self.chosen_1 = 0
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.candidates = [HonestCandidate(self.nissues, self.pvs) for _ in range(self.ncans - 1)]
        self.malicious = MaliciousCandidate(self.nissues, self.pvs)
        self.candidates.append(self.malicious)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        #print(self.diff)

        return [len(self.diff), self.malicious.honesty]




def main():
    # env = RD_voting_multi()
    # #check_env(env)
    # #vec_env = DummyVecEnv([lambda: FL_mnist()])
    # #vec_env = VecCheckNan(vec_env, raise_exception = True)
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    # checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="RD_multi_0.9pen_1.003regen_borda_1/",
    #                                          name_prefix='rl_model')
    # #model = TD3("MultiInputPolicy", env, buffer_size = 1000000,
    # #            policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="try_mnist_td3_fltrust_g_black/",
    # #            verbose=1, gamma = 1, action_noise = action_noise, learning_rate=linear_schedule(1e-6), learning_starts = 2000, train_freq = (5, "step"), batch_size = 512)
    #
    # model = SAC("MlpPolicy", env, buffer_size = 1000000,
    #             policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="RD_multi_0.9pen_1.003regen_borda_1/",
    #             verbose=1, gamma = 1, action_noise = action_noise)
    # model.learn(total_timesteps=500000, callback = checkpoint_callback)

    env = FRD_voting_multi()
    #check_env(env)
    #vec_env = DummyVecEnv([lambda: FL_mnist()])
    #vec_env = VecCheckNan(vec_env, raise_exception = True)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path="FRD_multi_0.9pen_1.003regen_weights/",
                                             name_prefix='rl_model')
    #model = TD3("MultiInputPolicy", env, buffer_size = 1000000,
    #            policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="try_mnist_td3_fltrust_g_black/",
    #            verbose=1, gamma = 1, action_noise = action_noise, learning_rate=linear_schedule(1e-6), learning_starts = 2000, train_freq = (5, "step"), batch_size = 512)

    model = SAC("MlpPolicy", env, buffer_size = 1000000,
                policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="FRD_multi_0.9pen_1.003regen_weights/",
                verbose=1, gamma = 1, action_noise = action_noise)
    model.learn(total_timesteps=200000, callback = checkpoint_callback)

if __name__ == '__main__':
    main()
#main()
