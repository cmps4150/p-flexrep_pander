
"""
Gives agreement between x and y (both numpy arrays!!)
"""
def agreement(x, y):
    return 1 - sum(abs(x - y))

def get_preferences_on_candidates(pvs, ppc, pmc):
    pvc = []
    for voter in pvs:
        agreements = [(agreement(voter, ppc[i]), i) for i in range(len(ppc))]
        agreements.append((agreement(voter, pmc), len(ppc)))
        agreements = agreements.sort(reverse=True)
        pvc.append(list(list(zip(*agreements))[1]))
    return pvc



"""
Get committe of size k from candidates using voting rule vr
Params:
    pvs - voter preference profile on issues (np array of size (num voters, num issues))
    ppc - public candidate preference profile (np array of size (num cans - 1, num issues))
    pmv - public preferences of malicious candidate (np array of size (num issues))
"""
def get_committee(pvs, ppc, pmc, vr, k):
    return vr(pvs, ppc, pmc, k)
