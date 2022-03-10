import numpy as np

"""
Gives agreement between x and y (both numpy arrays!!)
"""
def agreement(x, y):
    return 1 - sum(abs(x - y))

def num_agreement(x, y):
    return sum(abs(x-y))

"""
Gets preferences of voters on candidates by ordering them based on their agreement.
Params:
    pvs - voter preference profile on issues (np array of size (num voters, num issues))
    ppc - public candidate preference profile (np array of size (num cans - 1, num issues))
    pmc - public preferences of malicious candidate (np array of size (num issues))
"""
def get_preferences_on_candidates(pvs, ppc, pmc):
    # the ith entry is the ith voters ranking of candidates
    # so pvc[0][0] = (lowest agreement value, lowest agreement index)
    pvc = []
    # loop over voter preference profiles on votes
    for voter in pvs:
        # get the agreement of the voter with each candidate
        agreements = [(agreement(voter, ppc[i]), i) for i in range(len(ppc))]
        # get agreement with the malicious candidate
        agreements.append((agreement(voter, pmc), len(ppc)))
        # sort so that highest agreement is last
        agreements.sort()
        pvc.append(agreements)
    return np.array(pvc)

def get_approval_ballots(pvs, ppc, pmc):
    weights = []
    for voter in pvs:
        weights = [num_agreements(voter, ppc[i]) for i in range(len(ppc))]
        weights.append(num_agreements(voter, pmc))

    ballots = [(weights[i]/sum(weights), i) for i in range(len(weights))]
    return ballots


def borda(pvc):
    # create scores that are initially 0 for all candidates
    scores = [[0, i] for i in range(pvc.shape[1])]
    # loop over each ranking
    for ranking in pvc:
        # loop over the rankings, skip over 0 (no points in borda)
        for i in range(1, len(ranking)):
            # candidate at position i gets i points
            cand = ranking[i][1]
            scores[cand][0] += i
    scores.sort(reverse=True)
    # return the ordering of candidates based on scores
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
