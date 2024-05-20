import numpy as np

"""Term frequency normalization"""


# Relative Intra-document TF (RITF)
# @param tf: TF(t, D) -> term frequency of t in document D
# @param avgtf: Avg.TF(D) -> average term frequency of document D
# @param c: free parameter (default is 1)
def ritf(tf, avgtf, c=1):
    return np.log2(1 + tf) / np.log2(c + avgtf)


# Length regularized TF (LRTF)
# @param tf: TF(t, D) -> term frequency of t in document D
# @param adl: ADL(C) -> average document length in collection C
# @param dl: len(D) -> length of document D
def lrtf(tf, adl, dl):
    return tf * np.log2(1 + adl / dl)
