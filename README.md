# parafac_submission

Individual networks provided by the challenge were merged with C++ based on lemon (https://github.com/sebwink/dream_merge_graph)
The output "dream_merge.graphml" is then used for the PARAFAC method.

The analytical steps listed in parafac_submission.py were used with the python 3.5 interactive shell. It is assumed that "dream_merge.graphml" is present in the folder from which the interactive shell is called, the predicted disease modules are written as "nonneg_parafac_submission.gmt" in the same folder.
