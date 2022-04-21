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
        # sort so that highest agreement is last
        agreements.sort()
        pvc.append(agreements)
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

    def __init__(self, nissues, pvs):
        self.rng = default_rng()
        self.nissues = nissues
        self.private_profile = self.rng.choice([True, False], size=nissues)
        self.honesty = 1
        self.change = 0

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
        #print(changelist)
        self.public_profile = copy.deepcopy(self.private_profile)
        for i in changelist:
            if self.public_profile[i]:
                self.public_profile[i] = False
            else:
                self.public_profile[i] = True
            #self.public_profile[i] = -self.public_profile[i]

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
        self.vr = weights
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
        reward = agree * 10 + in_committe *3
        self.round += 1
        #next round
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        for can in self.candidates:
            can.private_profile = self.rng.choice([True, False], size=self.nissues)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)

        if self.round >= self.max_rnd:
            done = True
        else:
            done = False


        return [len(self.diff), self.malicious.honesty], reward, done, {}


    def reset(self):
        self.round = 0
        self.pvs = self.rng.choice([False, True], size=(self.nvoters, self.nissues))
        self.candidates = [HonestCandidate(self.nissues, self.pvs) for _ in range(self.ncans - 1)]
        self.malicious = MaliciousCandidate(self.nissues, self.pvs)
        self.candidates.append(self.malicious)
        self.diff = diff_public_attacker(self.nissues, self.pvs, self.malicious.private_profile, self.nvoters)
        #print(self.diff)

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

def main():
    env = RD_voting()
    #check_env(env)
    #vec_env = DummyVecEnv([lambda: FL_mnist()])
    #vec_env = VecCheckNan(vec_env, raise_exception = True)
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path="test/",
                                             name_prefix='rl_model')
    #model = TD3("MultiInputPolicy", env, buffer_size = 1000000,
    #            policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="try_mnist_td3_fltrust_g_black/",
    #            verbose=1, gamma = 1, action_noise = action_noise, learning_rate=linear_schedule(1e-6), learning_starts = 2000, train_freq = (5, "step"), batch_size = 512)

    model = SAC("MlpPolicy", env, buffer_size = 1000000,
                policy_kwargs={"net_arch" : [256,128]}, tensorboard_log="test/",
                verbose=1, gamma = 0.99, action_noise = action_noise)
    model.learn(total_timesteps=300000, callback = checkpoint_callback)

main()
