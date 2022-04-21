from numpy.random import default_rng
import gurobipy as gpy
from voting import *

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

    """
    Set the malicious candidate's public profile. Depends on the voters preference
    Parameters:
        m : the number of issues the voter changes to maximize the agreement
            between its private profile and the outcome
    """
    def set_public_profile(self, pvs, m):
        # get the
        differences = diff_public_attacker(nissues, pvs, ppc, nvoters)
        # deviate from private profile based on parameter `change`
        # action of malicious candidate
        # one round: given from results of MIP
        # multiround: given from RL results
        pass

class MIP:
    def __init__(ncans, k, nvoters, nissues, vr=weights, frd=False):
        self.rng = default_rng()
        self.ncans = ncans
        self.k = k  # committee size
        self.nvoters = nvoters
        self.nissues = nissues
        self.vr = vr
        # voter preference profile on issues
        self.pvs = self.rng.choice([False, True], size=(nvoters, nissues))

        self.candidates = [HonestCandidate(self.nissues) for _ in range(ncans - 1)]
        malicious = MaliciousCandidate(self.nissues)
        self.candidates.append(malicious)

        model = gpy.Model("Voting")
        # all issues that differ from the malicious candidate and public opinion
        model.addVars()

    def run_round():
        ppc = [can.public_profile for can in self.candidates]
        pvc = get_preferences_on_candidates(self.pvs, ppc)
        committee = get_committee(pvc, self.vr, self.k)
        if self.frd:
            self.outcomes = get_outcomes_frd(committee, self.nissues, self.pvs)
        else:
            self.outcomes = get_outcomes_rd(committee, self.nissues)
