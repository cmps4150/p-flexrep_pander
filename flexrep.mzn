include "globals.mzn";

int: nvoters;
int: ncans;
int: nreps;
int: nissues;

% voter preferences on issues
array[1..nvoters, 1..nissues] of bool: PVS;
PVS = array2d(1..nvoters, 1..nissues, [bernoulli(0.5) | i in 1..nissues, j in 1..nvoters]);
% candidate's private preferences on issues
array[1..ncans, 1..nissues] of bool: PCS;
PCS = array2d(1..ncans, 1..nissues, [bernoulli(0.5) | i in 1..nissues, j in 1..ncans]);
% scores for the candidates, CURRENTLY ONLY VALID FOR APPROVAL VOTING
array[1..ncans-1] of 0..nvoters: scores = [sum (j in 1..nvoters where agreement(PVS[j, 1..nissues], PCS[c, 1..nissues], nissues) > 0.5) (1) | c in 1..ncans-1];

% malicious candidate's public preference profile
array[1..nissues] of var bool: mcpub;
% all scores of the candidates
array[1..ncans] of var 0..nvoters: allscores;
% array which indicates whether candidate c is elected as a representative
array[1..ncans] of var bool: reps;
% array containing the outcomes of the issues
array[1..nissues] of var bool: outcomes;

constraint forall (c in 1..ncans-1) (allscores[c] == scores[c]);
% set the score for the malicious candidate
constraint allscores[ncans] == sum (j in 1..nvoters where agreement(PVS[j, 1..nissues], mcpub, nissues) > 0.5) (1);

% ensure that we only elect nreps representatives
constraint sum (c in 1..ncans where reps[c]) (1) == nreps;
% ensure that elected representatives have a score at least as high as other candidates
constraint forall (c in 1..ncans where reps[c]) (forall (c1 in 1..ncans where not reps[c]) (allscores[c] >= allscores[c1]));

% get outcomes from private candidate preferences
constraint forall (i in 1..nissues) (outcomes[i] == (sum (c in 1..ncans where reps[c] /\ PCS[c, i]) (1) > (nreps / 2) ) );

% normal agreement function for two arrays with no decision variables
function float: agreement(array[int] of bool: a, array[int] of bool: b, int: ni) =
  1 - (1 / ni * sum (i in 1..ni where a[i] != b[i]) (1));
% agreement function where the first array contains decision variables, and the second does not
function var float: agreement(array[int] of var bool: a, array[int] of bool: b, int: ni) =
  1 - (1 / ni * sum (i in 1..ni where a[i] != b[i]) (1));
% agreement function where both arrays contain decision variables, and the second does not
function var float: agreement(array[int] of var bool: a, array[int] of var bool: b, int: ni) =
  1 - (1 / ni * sum (i in 1..ni where a[i] != b[i]) (1));

% maximize agreement between PCS[-1] (the malicious candidates private preference profile) and outcome
solve maximize agreement(PCS[ncans, 1..nissues], outcomes, nissues);

output ["Agreement between public and private candidate preferences: \(agreement(mcpub, PCS[ncans, 1..nissues], nissues))\n",
         "Public candidate preference profile:", show(mcpub), "\n",
         "Private candidate preference profile:", show(PCS[ncans, 1..nissues]), "\n",
         "Outcomes:", show(outcomes), "\n"];