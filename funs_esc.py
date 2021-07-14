import pandas as pd

# Script for the escalation functions
ymi = -1000
ymx = +1000

esc_bins = [ymi, 31, 38, 48, ymx]
esc_lbls = ['≤30', '31-37', '38-47', '≥48']

"""
x: pandas data.frame with at least columns cn
cn: the columns for which escalation bins should be calculated for
esc_bins: list of bins
esc_lbls: the matching labels (length-1 of esc_bins)
    returns
    x with columns "esc_"+cn
"""
def get_esc_levels(x, cn, esc_bins, esc_lbls):
    # Function to calculate escalation levels for different columns
    assert isinstance(x,pd.DataFrame)
    if not isinstance(cn, pd.Series):
        cn = pd.Series(cn)
    cn2 = 'esc_' + cn
    z = x[cn].apply(lambda w: pd.Categorical(pd.cut(w, esc_bins, False, esc_lbls)).codes,0)
    z.rename(columns=dict(zip(cn,cn2)),inplace=True)
    return pd.concat([x,z],axis=1)
