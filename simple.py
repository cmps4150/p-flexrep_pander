from numpy.random import default_rng
import numpy as np
import gurobipy as gpy
from gurobipy import GRB

rng = default_rng()

nvoters = 10
ncans = 10
nreps = 3
nissues = 10
# what voting rule to use, currently, only approval voting is allowed
vr = "av"
# whether to use rd or frd
rd = True

m = gpy.Model("Voting")
# malicious candidate public preference profile
mcpub = m.addVars(nissues, vtype=GRB.BINARY)
# issue outcomes
outcomes = m.addVars(nissues, vtype=GRB.BINARY)

# whether voter j disagrees with the malicious candidate on issue i
mdisagree = m.addVars(nvoters, nissues, vtype=GRB.BINARY)
# agreement values of voter j with the malicious candidate
magreements = m.addVars(nvoters, vtype=GRB.CONTINUOUS)

# voter preference profile on issues
pvs = rng.choice([0, 1], size=(nvoters, nissues))

# private preferences of candidates on issues
pcs = rng.choice([0, 1], size=(ncans, nissues))

# agreement of all voters with all honest candidates, (nvoters, ncans-1)
agreements = np.array([[1 - (1 / nissues) * sum(pvs[j, i] ^ pcs[c, i] for i in range(nissues))
                for c in range(ncans - 1)] for j in range(nvoters)])
# set disagreements between each voter and the malicious candidate
mdisagree_temp = m.addVars(nvoters, nissues, vtype=GRB.BINARY)
m.addConstrs(mdisagree_temp[j, i] == (pvs[j,i] - mcpub[i]) for j in range(nvoters)
                for i in range(nissues))
m.addConstrs(mdisagree[j, i] == gpy.abs_(mdisagree_temp[j, i]) for j in range(nvoters)
                for i in range(nissues))
# set agreement values for each voter with the malicious candidate
m.addConstrs((1 - (1 / nissues) * sum(mdisagree[j, i] for i in range(nissues))) == magreements[j]
                for j in range(nvoters))

if vr == "av":
    # whether voter j approves of candidate c
    approvals = np.array([[agreements[j, c] > .5 for c in range(ncans - 1)] for j in range(nvoters)])
    # whether voter j approves of the malicious candidate
    mapprovals = m.addVars(nvoters, vtype=GRB.BINARY)
    # set malicious voter approvals
    for j in range(nvoters):
        m.addGenConstrPWL(magreements[j], mapprovals[j], [0, 0.5, 0.5, 1], [0, 0, 1, 1])
    # get scores for candidates according to av
    scores = [(sum(approvals[j, c] for j in range(nvoters)), c) for c in range(ncans - 1)]
    mscore = m.addVar(vtype=GRB.INTEGER)
    m.addConstr(sum(mapprovals[j] for j in range(nvoters)) == mscore)

    # sort scores from highest to lowest
    scores.sort(reverse=True)
    top_cands = scores[:nreps]
    top_inds = list(zip(*top_cands))[1]
    # indicates if the malicious candidate becomes a representative
    isrep = m.addVar(vtype=GRB.BINARY)
    # malicious candidate becomes a representative if their approval score is higher
    # than the lowest score in top reps
    m.addGenConstrPWL(mscore, isrep, [0, top_cands[-1][0] - 1, top_cands[-1][0], nvoters + 1], [0, 0, 1, 1])
    # indicates whether a candidate is a representative or not
    reps = m.addVars(ncans, GRB.BINARY)
    m.addConstr(reps[ncans-1, 'B'] == isrep)
    m.addConstr(reps[top_inds[-1], 'B'] == (not isrep))
    for c in range(ncans - 1):
        if c != top_inds[-1]:
            m.addConstr(reps[c, 'B'] == (c in top_inds))
else:
    print("Undefined voting rule: {}".format(vr))

if rd:
    outcomes = m.addVars(nissues, vtype=GRB.BINARY)
    m.addConstrs(((1 / nreps) * sum(pcs[c, i] for c in range(ncans)) >= .5) == outcomes[i] for i in range(nissues))
else:
    print("We have not implemented flexible representative democracy yet")

# whether an outcome differs from the malicious candidates private preference profile
outcomes_diff = m.addVars(nissues, vtype=GRB.BINARY)
outcomes_diff_temp = m.addVars(nissues, vtype=GRB.BINARY)
m.addConstrs(outcomes_diff_temp[i] == (outcomes[i] - pcs[-1, i]) for i in range(nissues))
m.addConstrs(outcomes_diff[i] == gpy.abs_(outcomes_diff_temp[i]) for i in range(nissues))
m.setObjective(1 - (1 / nissues) * sum(outcomes_diff[i] for i in range(nissues)), GRB.MAXIMIZE)

m.optimize()

print("Malicious candidate's private preference profile:", [mcpub[i].X for i in range(nissues)])
print(" Malicious candidate's public preference profile:", [pcs[-1, i] for i in range(nissues)])
print("                                        Outcomes:", [outcomes[i].X for i in range(nissues)])
