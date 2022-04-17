import numpy as np
from numpy.random import default_rng

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

    def set_public_profile(self, pvs):
        # get popular positions on issues
        # deviate from private profile based on parameter `change`
        # action of malicious candidate
        # one round: given from results of MIP
        # multiround: given from RL results
        pass


class HonestCandidate(Candidate):
    def set_public_profile(self, pvs):
        self.public_profile = self.private_profile


"""
Gives agreement between x and y (both numpy arrays!!)
"""
def agreement(x, y):
    assert len(x) == len(y)
    return 1 - (1/len(x)) * sum(x ^ y)

def num_agreement(x, y):
    return sum(abs(x-y))

"""
Gets preferences of voters on candidates by ordering them based on their agreement.
Params:
    pvs - voter preference profile on issues (np array of size (num voters, num issues))
    ppc - public candidate preference profile (np array of size (num cans, num issues))
"""
def get_preferences_on_candidates(pvs, ppc):
    # the ith entry is the ith voters ranking of candidates, so
    # pvc[i][0] = (lowest agreement value, lowest agreement index) for voter i
    pvc = []
    # loop over voter preference profiles on votes
    for voter in pvs:
        # get the agreement of the voter with each candidate
        agreements = [(agreement(voter, ppc[i]), i + 1) for i in range(len(ppc))]
        # sort so that highest agreement is last
        agreements.sort()
        pvc.append(agreements)
    return np.array(pvc)

"""
Borda voting rule
"""
def borda(pvc):
    # create scores that are initially 0 for all candidates
    scores = [[0, i + 1] for i in range(pvc.shape[1])]
    # loop over each ranking
    for ranking in pvc:
        # loop over the rankings, skip over 0 (no points in borda)
        for i in range(1, len(ranking)):
            # candidate at position i gets i points
            cand = int(ranking[i][1])
            scores[cand - 1][0] += i
    scores.sort(reverse=True)
    # return the ordering of candidates based on scores
    return [x[1] for x in scores]

"""
Single Tranferable Vote election returns ordering of candidates
"""
def stv(pvc):
    num_cands = pvc.shape[1]
    # candidates who have been eliminated
    eliminated = []
    for cycle in range(num_cands):
        # create scores that are initially 0 for all candidates
        scores = [[0, i + 1] for i in range(num_cands)]
        # loop over each ranking
        for ranking in pvc:
            # the last ranking is the top one
            i = -1
            # if candidate is eliminated, go to the next one
            while int(ranking[i][1]) in eliminated:
                i -= 1
            # candidate with highest rank in each profile gets a vote
            cand = int(ranking[i][1])
            scores[cand - 1][0] += 1
        scores.sort()
        # eliminate the candidate who is last in this cycle
        eliminated.append(scores[cycle][1])
    eliminated.reverse()
    # return the ordering of candidates based on order of elimination
    return eliminated

"""
Chamberlain-Courant rule
(IS THIS CORRECT, OR SHOULD SCORE BE INCREASED BY SOME OTHER NUMBER?)
"""
def cc(pvc):
    # create scores that are initially 0 for all candidates
    scores = [[0, i + 1] for i in range(pvc.shape[1])]
    # loop over each ranking
    for ranking in pvc:
        # get candidate with highest ranking
        cand = int(ranking[-1][1])
        scores[cand - 1][0] += ranking[-1][0]
    scores.sort(reverse=True)
    print(scores)
    # return the ordering of candidates based on scores
    return [x[1] for x in scores]

"""
k-median rule
Parameters:
    med - the index (starting from 1) of the candidate whose score we consider
(IS THIS CORRECT, OR SHOULD SCORE BE INCREASED BY SOME OTHER NUMBER?)
The cited paper says that we maximize utlity, but FRD paper says this is used
when candidates submit total preference orderings
"""
def kmed(pvc, med):
    # create scores that are initially 0 for all candidates
    scores = [[0, i + 1] for i in range(pvc.shape[1])]
    # loop over each ranking
    for ranking in pvc:
        # get candidate with the medth highest ranking
        cand = int(ranking[-med][1])
        scores[cand - 1][0] += ranking[-med][0]
    scores.sort(reverse=True)
    # return the ordering of candidates based on scores
    return [x[1] for x in scores]


"""
Approval Voting
"""
def av(pvc):
    # create scores that are initially 0 for all candidates
    scores = [[0, i + 1] for i in range(pvc.shape[1])]
    # loop over each ranking
    for ranking in pvc:
        # loop over each candidate
        for i in range(len(ranking)):
            # give score of 1 if the candidate has more than 1/2 agreement with
            # the voter who submitted this ranking
            if ranking[i][0] > .5:
                cand = int(ranking[i][1])
                scores[cand - 1][0] += 1
    scores.sort(reverse=True)
    print(scores)
    # return the ordering of candidates based on scores
    return [x[1] for x in scores]

"""
Reweighted Approval Voting
"""
def rav(pvc):
    num_cands = pvc.shape[1]
    # contains all elected candidates
    elected = []
    for round in range(1, num_cands + 1):
        # create scores that are initially 0 for all candidates
        scores = [[0, i + 1] for i in range(pvc.shape[1])]
        # loop over each ranking
        for ranking in pvc:
            # factor by which to reweight the approval ballot
            factor = 1
            # a list which contains all of the candidates this voter approves of
            approvals = []
            # loop over each candidate
            for i in range(len(ranking)):
                # if the candidate has more than 1/2 agreement with the voter
                # who submitted this ranking, they are approved of
                if ranking[i][0] > .5:
                    cand = int(ranking[i][1])
                    approvals.append(cand)
                    # increase the factor by 1 for each candidate that has already
                    # been elected
                    if cand in elected:
                        factor += 1
            # give candidates their scores reweighted by 1/factor
            for cand in approvals:
                scores[cand - 1][0] += 1/factor
        # get the winner (last index)
        scores.sort()
        i = -1
        # only add a winner who has not yet been elected
        while scores[i][1] in elected:
            i -= 1
        elected.append(scores[i][1])
    return elected

"""
Weights Agreement
"""
def weights(pvc):
    # create scores that are initially 0 for all candidates
    scores = [[0, i + 1] for i in range(pvc.shape[1])]
    for ranking in pvc:
        # get the sum of all agreements for this voter
        tot = sum([ranking[i][0] for i in range(len(ranking))])
        for i in range(len(ranking)):
            # increase score by normalized weight
            cand = int(ranking[i][1])
            scores[cand - 1][0] += ranking[i][0] / tot
    scores.sort(reverse=True)
    print(scores)
    return [x[1] for x in scores]

"""
Get committe of size k from candidates using voting rule vr
Params:
    pvc - voter preference profile on candidates (np array of size (num voters, num cans, 2))
MAY HAVE TO CHANGE THIS BASED ON VOTING RULE
"""
def get_committee(pvc, vr, k):
    scores = vr(pvc)
    return scores[:k]


"""
Get the outcome of an election in a representative democracy
Params:
    committee - list of Candidate objects
    nissues - the number of issues under consideration
"""
def get_outcomes_rd(committee, nissues):
    outcomes = []
    for i in range(nissues):
        yes = sum([can.private_profile[i] for can in commitee]) >= len(committee)//2 #True if majority of candidates voted 1 for issue
        # update malicious candidate's honesty if deviate
        for can in commitee:
            if can.private_profile[i] != can.public_profile[i]:
                # Multiple choices: exponential decay / constant decay
                can.honesty = can.honesty * 0.8
                # can.honesty = max(can.honesty - 0.1, 0)
            #honesty recovers a little bit if not deviate
            else:
                can.honesty = can.honesty *1.05
        outcomes.append(yes)
    return np.array(outcomes)

"""
Get the outcome of an election in a flexible representative democracy
Params:
    committee - list of Candidate objects
    nissues - the number of issues under consideration
    pvs - voters preference on issues
"""
def get_outcomes_frd(committee, nissues, pvs):
    rng = default_rng()
    # outcomes is a list of booleans with the length of the issues
    outcomes = np.zeros(size=nissues)
    for i in range(nissues):
        # weights is a list of weights that voters delegate to each candidate for each issue
        weights = np.zeros(size=len(committee))
        for voter in pvs:
            yes = voter[i]  # voter's opinion on issue i

            #see which candidates the voter will delegate to (if any) based on agreement
            delegate_to = []
            for c in range(len(committee)):
                can = commitee[c]
                # if they agree on the issue
                if yes == can.private_profile[i]:
                    if rng.choice([True, False], p=[alpha * can.honesty, 1 - (alpha * can.honesty)]):
                        delegate_to.append(c)
                # update malicious candidate's honesty if deviate
                if can.private_profile[i] != can.public_profile[i]:
                    # Multiple choices: exponential decay / constant decay
                    can.honesty = can.honesty * 0.8
                    # can.honesty = max(can.honesty - 0.1, 0)
                #honesty recovers a little bit if not deviate
                else:
                    can.honesty = can.honesty *1.05
            # default distribution mechanism, equal weight to all committee members
            if len(delegate_to) == 0:
                weights += 1/len(committee)
            #otherwise, weights distributed over candidates the voter delegates to
            else:
                for j in delegate_to:
                    weights[j] += 1/(len(delegate_to))
        # total weight of committee members who vote yes on issue i
        yes_weight = 0
        for c in len(committee):
            if committee[c].private_profile[i]:
                yes_weight += weights[c]
        outcomes[i] = yes_weight / len(pvs) >= .5
    return outcomes
