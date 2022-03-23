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
    # the ith entry is the ith voters ranking of candidates, so
    # pvc[i][0] = (lowest agreement value, lowest agreement index) for voter i
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
