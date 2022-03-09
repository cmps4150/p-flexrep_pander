from numpy.random import default_rng

rng = default_rng()

nvoters = 100
ncans = 30
nreps = 10
nissues = 20

# voter preference profile on issues
pvs = rng.choice([False, True], size=(nvoters, nissues))

# private candidate preference profile
prpc = rng.choice([False, True], size=(ncans, nissues))

# public candidate preference profile
# (removes one because one candidate is malicious)
ppc = prpc[:-1]

# public preferences for malicious candidate
pmc = rng.choice([False, True], size=nissues)
