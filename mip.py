Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore

@jnmasur
cmps4150
/
p-flexrep
Private
Code
Issues
Pull requests
Actions
Projects
Security
Insights
Settings
p-flexrep/mip.py /
@SliencerX
SliencerX RL runs, fix bugs
Latest commit daf0ed6 4 days ago
 History
 2 contributors
@SliencerX@jnmasur
68 lines (55 sloc)  2.46 KB

from numpy.random import default_rng
import gurobipy as gpy
from gurobipy import GRB

class MaliciousCandidate(Candidate):
    """
    Defines a malicious (pandering) candidate
    """

    """
    Set the malicious candidate's public profile. Depends on the voters preference
    Parameters:
        m : the number of issues the voter changes to maximize the agreement
            between its private profile and the outcome
    """
    def set_public_profile(self, differences, m):
        # get the issues where the attacker differs from public opinion
        differences = differences[:m] #assuming the candidate switches on issues for which they differ most
        public_profile = self.private_profile
        diff_inds = [x[0] for x in differences]
        for ind in diff_inds:
            public_profile[ind] = not public_profile[ind]

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

        differences = diff_public_attacker(self.nissues, self.pvs, self.candidates[-1].private_profile, self.nvoters)

        model = gpy.Model("Voting")
        # all issues that differ from the malicious candidate and public opinion
        m = model.addVar(ub=self.nissues, vtype=GRB.INTEGER)

        model.setObjective()



    def run_round():
        ppc = [can.public_profile for can in self.candidates]
        pvc = get_preferences_on_candidates(self.pvs, ppc)
        committee = get_committee(pvc, self.vr, self.k)
        if self.frd:
            self.outcomes = get_outcomes_frd(committee, self.nissues, self.pvs)
        else:
            self.outcomes = get_outcomes_rd(committee, self.nissues)

constant = sum(outcome[i] - private_profile[i] for i not in diff_inds)
model.setObjective(1 - 1/N * (sum(outcome[i] - private_profile[i] for i in diff_inds)) + constant)
outcome = [sum(delegate_preferences_on_issues[i]) for i in range(num_issues)]
delegates =

"""
M = model.addVars(N, vtype=GRB.BINARY)
model.setObjective(1- (1/N) * sum(outcome[i] - M[i])
diff(pref, M) -> how many deviated
"""
