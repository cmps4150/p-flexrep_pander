from numpy.random import default_rng
import gurobipy as gpy

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
        self.candidates.append()

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
