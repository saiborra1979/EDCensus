import re
import numpy as np
import pandas as pd
import itertools
from math import radians, cos, sin, asin, sqrt

def add_lags(df,l):
    tmp = df.shift(l)
    tmp.columns = pd.MultiIndex.from_product([tmp.columns,['lag_'+str(l)]])
    return tmp

def ljoin(x):
    return list(itertools.chain.from_iterable(x))

# Function to calculate haversine distance between two points
def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def vhaversine(lat1, lon1, lat2, lon2):
    fun = np.vectorize(haversine)
    vals = fun(lat1, lon1, lat2, lon2)
    return (vals)

def pc_extract(ss, pat):
    hit = re.search(pat, ss)
    if hit is None:
        return 0
    else:
        return hit.end()