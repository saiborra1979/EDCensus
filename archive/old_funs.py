from sklearn.metrics import r2_score as r2
from sklearn.metrics import mean_squared_error as mse

def get_perf(groups):
    cn = groups[0]
    df = groups[1]
    res = pd.DataFrame(cn).T.assign(rmse=mse(df.y, df.pred, squared=False), r2=r2(df.y, df.pred), conc=cindex(df.y, df.pred))
    return res

# data=res_rest.copy();gg=cn_multi;n_cpus=10
def parallel_perf(data, gg, n_cpus=None):
    data_split = data.groupby(gg)
    if n_cpus is None:
        n_cpus = os.cpu_count()-1
    print('Number of CPUs: %i' % n_cpus)
    pool = multiprocessing.Pool(processes=n_cpus)
    data = pd.concat(pool.map(get_perf, data_split))
    pool.close()
    pool.join()
    return data

