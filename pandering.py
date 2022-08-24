'''
Compute pandering strategies for strategic representatives
'''
import numpy as np
import pandas as pd

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr


def optimal_pander(profile, max_diff_for_approval, true_pref, max_issues_pandered):

    return session.evaluate(f'maximizeApprovals[{profile}, {max_diff_for_approval}, {true_pref}, {max_issues_pandered}]')


if __name__ == '__main__':

    #Start the session and get the WL functions running before calling them
    session = WolframLanguageSession()

    session.evaluate('binVec = {(0 | 1) ..};')
    session.evaluate('distanceFunction[a : binVec][b : binVec] := HammingDistance[a, b];')
    session.evaluate('numApprovals[x : {binVec ..}, ys : binVec, k_] := Length[Select[Map[distanceFunction[ys], x], (# <= k) &]];')

    #version without constraint, but with optional penalty
    session.evaluate('maximizeApprovals[x : {binVec ..}, k_, penalty_ : (0 &)] := \
      Module[{y}, \
       With[{ys = Array[y, Length[First[x]]]}, \
        Maximize[{numApprovals[x, ys, k] + penalty[ys], \
           Thread[0 <= ys <= 1]}, \
          ys \[Element] Integers] // {First[#], ys /. Last[#]} &]];')

    #version with constraint, via suitably large penalty
    session.evaluate('maximizeApprovals[x_, k_, ref_, j_] := \
      With[{p = -Length[x] - 1}, \
       maximizeApprovals[x, k, \
        If[distanceFunction[ref][#] <= j, 0, p] &]];')\

    #Call the functions here for optimal pandering
    #print(optimal_pander('{{1, 1, 1}, {0, 1, 0}, {1, 0, 0}}', 0))
    #print(optimal_pander('{{1, 1, 1}, {0, 1, 0}, {1, 0, 0}}', 1)) # (3,(1,1,0))
    #print(optimal_pander('{{1, 1, 1}, {0, 1, 0}, {1, 0, 0}}', 2))

    print("Test profile: {} with true rep preferences {}".format('{{1, 1, 1}, {0, 1, 0}, {1, 0, 0}}', '{0,0,0}'))
    print("Agreement on at least half of issues required for agreement (2/3)")
    print("No pandering: {}".format(optimal_pander('{{1, 1, 1}, {0, 1, 0}, {1, 0, 0}}', 1, '{0, 0, 0}', 0)))
    print("Max 1 issue pandered: {}".format(optimal_pander('{{1, 1, 1}, {0, 1, 0}, {1, 0, 0}}', 1, '{0, 0, 0}', 1))) # (2,(0,0,0))
    print("Max 2 issues pandered: {}".format(optimal_pander('{{1, 1, 1}, {0, 1, 0}, {1, 0, 0}}', 1, '{0, 0, 0}', 2)))


    #Need to terminate the session after all the instances are run
    session.terminate()


