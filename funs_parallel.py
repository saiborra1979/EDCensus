import os
import pandas as pd
import numpy as np
import multiprocessing

def get_perf_month(groups):
    cn = groups[0]
    df = groups[1]
    res1 = pd.DataFrame({'n':df.n.sum(), 'value':df.value.mean(),
            'se':np.sqrt( np.sum(df.se**2 / df.n)) },index=[0])
    res2 = df.head(1).drop(columns=['n','value','se','month']).reset_index(None,True)
    res = pd.concat([res2, res1],axis=1)
    return res

# data=res_rest.copy();gg=cn_multi;n_cpus=10
def parallel_perf(data, gg, fun, n_cpus=None):
    data_split = data.groupby(gg)
    if n_cpus is None:
        n_cpus = max(os.cpu_count()-1, 1)
    print('Number of CPUs: %i' % n_cpus)
    pool = multiprocessing.Pool(processes=n_cpus)
    res = pd.concat(pool.map(fun, data_split))
    pool.close()
    pool.join()
    return res