import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth

def fpr(tp_all, tn_all, fp_all, fn_all, tp_grp, tn_grp, fp_grp, fn_grp):
    fpr_all = fp_all / (fp_all + tn_all)
    fpr_grp = fp_grp / (fp_grp + tn_grp)

    delta_fpr = fpr_grp - fpr_all
    return fpr_grp, delta_fpr

def error(tp_all, tn_all, fp_all, fn_all, tp_grp, tn_grp, fp_grp, fn_grp):
    error_all = (fn_all + fp_all) / (fp_all + tn_all + tp_all + fn_all)
    error_grp = (fn_grp + fp_grp) / (fp_grp + tn_grp + tp_grp + fn_grp)

    delta_error = error_grp - error_all
    return error_grp, delta_error

def tpr(tp_all, tn_all, fp_all, fn_all, tp_grp, tn_grp, fp_grp, fn_grp):
    tpr_all = tp_all / (tp_all + fn_all)
    tpr_grp = tp_grp / (tp_grp + fn_grp)

    delta_fpr = tpr_grp - tpr_all
    return tpr_grp, delta_fpr

def fnr(tp_all, tn_all, fp_all, fn_all, tp_grp, tn_grp, fp_grp, fn_grp):
    fnr_all = fn_all / (fn_all + tp_all)
    fnr_grp = fn_grp / (fn_grp + tp_grp)

    delta_fnr = fnr_grp - fnr_all
    return fnr_grp, delta_fnr

def tnr(tp_all, tn_all, fp_all, fn_all, tp_grp, tn_grp, fp_grp, fn_grp):
    tnr_all = tn_all / (tn_all + fp_all)
    tnr_grp = tn_grp / (tn_grp + fp_grp)

    delta_tnr = tnr_grp - tnr_all
    return tnr_grp, delta_tnr


def fpr_eff(conf_matrix_all, conf_matrix_grp):
    # tp tn fp fn
    fp_all = conf_matrix_all[2]
    tn_all = conf_matrix_all[1]
    fp_grp = conf_matrix_grp[2]
    tn_grp = conf_matrix_grp[1]

    fpr_all = fp_all / (fp_all + tn_all)
    fpr_grp = fp_grp / (fp_grp + tn_grp)

    delta_fpr = fpr_grp - fpr_all
    return fpr_grp.flatten(), delta_fpr.flatten()
def tpr_eff(conf_matrix_all, conf_matrix_grp):
    # tp tn fp fn
    tp_all = conf_matrix_all[0]
    fn_all = conf_matrix_all[3]
    tp_grp = conf_matrix_grp[0]
    fn_grp = conf_matrix_grp[3]

    tpr_all = tp_all / (tp_all + fn_all)
    tpr_grp = tp_grp / (tp_grp + fn_grp)

    delta_tpr = tpr_grp - tpr_all
    return tpr_grp.flatten(), delta_tpr.flatten()

def error_eff(conf_matrix_all, conf_matrix_grp):
    # tp tn fp fn
    tp_all, tn_all, fp_all, fn_all = conf_matrix_all
    tp_grp, tn_grp, fp_grp, fn_grp = conf_matrix_grp

    error_all = (fn_all + fp_all) / (fp_all + tn_all + tp_all + fn_all)
    error_grp = (fn_grp + fp_grp) / (fp_grp + tn_grp + tp_grp + fn_grp)

    delta_error = error_grp - error_all
    return error_grp.flatten(), delta_error.flatten()

def fnr_eff(conf_matrix_all, conf_matrix_grp):
    # tp tn fp fn
    tp_all = conf_matrix_all[0]
    fn_all = conf_matrix_all[3]
    tp_grp = conf_matrix_grp[0]
    fn_grp = conf_matrix_grp[3]

    fnr_all = fn_all / (fn_all + tp_all)
    fnr_grp = fn_grp / (fn_grp + tp_grp)

    delta_fnr = fnr_grp - fnr_all
    return fnr_grp.flatten(), delta_fnr.flatten()

def tnr_eff(conf_matrix_all, conf_matrix_grp):
    # tp tn fp fn
    tn_all = conf_matrix_all[1]
    fp_all = conf_matrix_all[2]
    tn_grp = conf_matrix_grp[1]
    fp_grp = conf_matrix_grp[2]

    tnr_all = tn_all / (tn_all + fp_all)
    tnr_grp = tn_grp / (tn_grp + fp_grp)

    delta_tnr = tnr_grp - tnr_all
    return tnr_grp.flatten(), delta_tnr.flatten()

from time import time

from collections import namedtuple
from scipy.sparse import csr_array, lil_array

Matches = namedtuple("Matches", ["matches", "fi"])


from multiprocess import pool
from time import time
from scipy.sparse import hstack as hstack_sparse

def compute_matches(df, fi=None, minsup=None, n_proc=18, chunk_size=400):
    if df is None and fi is None:
        raise ValueError("Either df or fi must be provided!")

    if fi is None:
        fi = fpgrowth(df, minsup)

    a = time()
    groups = lil_array((len(fi), len(df.columns)))
    for i, row in enumerate(fi.itemsets):
        # 1 / len(row) so that in `matches` the sum
        # will be 1 if all the items are present
        # (assuming that all points have value 1)
        groups[i, list(row)] = 1 / len(row)
    groups = csr_array(groups) # convert to csr -- improves performance!
    points = csr_array(df.values)

    def parallelizer(points, groups, chunk_size):
        def func(i_from):
            mm = points @ groups[i_from:i_from + chunk_size].T
            mm.data = mm.data >= (1 - 1e-8)
            mm.eliminate_zeros()
            return mm
        return func

    # parallelized computation of matches
    # (by computing the matches in chunks, we 
    # can re-sparsify each chunk, which is much faster)
    with pool.Pool(n_proc) as p:
        res = p.map(parallelizer(points, groups, chunk_size), range(0, groups.shape[0], chunk_size))
        mat = csr_array(hstack_sparse(res))

    # astype(int) ==> removed because it was using too much space
    # this needs to be converted to `int` when used
    return Matches(matches=mat, fi=fi)

def div_explorer(matches_obj, y_true, y_pred, metrics_list):
# def div_explorer(df, matches_obj, y_true, y_pred, metrics_list, minsup=None):
    # compute frequent itemsets, if needed
    # if matches_obj is None:
    #     matches_obj = compute_matches(df, minsup=minsup)

    from time import time
    matches, fi = matches_obj.matches, matches_obj.fi
    conf_matrix = np.vstack([((y_true == y_pred) & (y_pred == 1)), \
                  ((y_true == y_pred) & (y_pred == 0)), \
                  ((y_true != y_pred) & (y_pred == 1)), \
                  ((y_true != y_pred) & (y_pred == 0))])
    conf_matrix_all = conf_matrix.sum(axis=1) # tp_all, tn_all, fp_all, fn_all
    conf_matrix_grp = conf_matrix @ matches
    
    known_metrics = {
        "fpr": fpr_eff,
        "error": error_eff,
        "tpr": tpr_eff,
        "fnr": fnr_eff,
        "tnr": tnr_eff,
    }
    metrics_list = [ m for m in metrics_list if m in known_metrics ]
    
    metrics = { "subgroup": fi.itemsets }

    # disable warnings for 0 divisions (will be handled here)
    np.seterr(divide='ignore', invalid='ignore')
    for i, metric in enumerate(metrics_list):
        values, deltas = known_metrics[metric](conf_matrix_all, conf_matrix_grp)
        metrics[metric] = values
        metrics[f"d_{metric}"] = deltas
    # restore warnings
    np.seterr(divide='warn', invalid='warn')

    metrics["tp"] = conf_matrix_grp[0]
    metrics["tn"] = conf_matrix_grp[1]
    metrics["fp"] = conf_matrix_grp[2]
    metrics["fn"] = conf_matrix_grp[3]
    
    return pd.DataFrame(metrics)

def subgroups_1hot(metrics, df):
    # df assumed 1hot
    subgroups = np.zeros((len(metrics), len(df.columns)), dtype=int)
    for i, sg in enumerate(metrics.subgroup):
        subgroups[i, list(sg)] = 1
    
    return subgroups


# returns a weight for each point
def get_weights(df, matches, metrics, metric, use_abs=False, aggregation="max", fill_policy="zero"):
    # metrics = div_explorer(df, matches, y_true, y_pred, [metric])

    # A = df.values[:, np.newaxis, :] # all points
    # B = subgroups_1hot(metrics, df) # all itemsets

    # matches = ((A & B) == B).all(axis=2) # all matches (i,j => whether point i contains itemset j)

    matches = matches.matches.todense()

    weights = np.zeros(len(df))

    for pt in range(len(df)):
        subgroups = metrics.iloc[matches[pt]][f"d_{metric}"]
        if use_abs:
            subgroups = subgroups.abs()
        if aggregation == "mean":
            weights[pt] = subgroups.mean()
        elif aggregation == "max":
            weights[pt] = subgroups.max()
        elif aggregation == "min":
            weights[pt] = subgroups.min()
        elif aggregation == "sum":
            weights[pt] = subgroups.sum()
        elif aggregation == "median":
            weights[pt] = subgroups.median()
        elif aggregation == "std":
            weights[pt] = subgroups.std()
        elif aggregation == "count":
            weights[pt] = subgroups.count()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    if fill_policy == "zero":
        weights = np.nan_to_num(weights, nan=0)
    elif fill_policy == "mean":
        weights = np.nan_to_num(weights, nan=weights.mean())
    elif fill_policy == "min":
        weights = np.nan_to_num(weights, nan=np.nanmin(weights))

    return weights

# returns a matrix of matches (i.e. which subgroups a point belongs to
def get_matches(df, fi, y_true, y_pred, metric):
    metrics = div_explorer(df, fi, y_true, y_pred, [metric])

    A = df.values[:, np.newaxis, :] # all points
    B = subgroups_1hot(metrics, df) # all itemsets

    matches = ((A & B) == B).all(axis=2) # all matches (i,j => whether point i contains itemset j)
    return matches