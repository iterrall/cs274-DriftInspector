import numpy as np
from functools import reduce


# NOTE: DEPRECATED (?!)
def detect_multibatch(divs, metric, ref_win, curr_win):

    start_ref_win, ref_win_size = ref_win
    start_curr_win, curr_win_size = curr_win

    eps = 1e-12

    E1 = reduce(lambda a,b : a.fillna(0) + b.fillna(0), [ div[metric] for div in divs[start_ref_win:start_ref_win+ref_win_size] ]) / ref_win_size
    E2 = reduce(lambda a,b : a.fillna(0) + b.fillna(0), [ div[metric] for div in divs[start_curr_win:start_curr_win+curr_win_size] ]) / curr_win_size

    delta = E2 - E1
    Var1 = reduce(lambda a,b : a + b, [ (div[metric] - E1) ** 2 for div in divs[start_ref_win:start_ref_win+ref_win_size] ]) / (ref_win_size - 1)
    Var2 = reduce(lambda a,b : a + b, [ (div[metric] - E2) ** 2 for div in divs[start_curr_win:start_curr_win+curr_win_size] ]) / (curr_win_size - 1)

    variance = Var1 / ref_win_size + Var2 / curr_win_size

    t_stat = abs(delta) / np.sqrt(variance + eps)

    return delta, t_stat

# compute the t-statistic & delta by considering
# multiple batches of adult.data as being a single, large one
# (this requires computing the metric for each window --
# as such it needs to be specified)
def detect_singlebatch(divs, metric, ref_win, curr_win):

    start_ref_win, ref_win_size = ref_win
    start_curr_win, curr_win_size = curr_win
    
    eps = 1e-12
    
    if metric == "accuracy":
        a1 = reduce(lambda a,b : a + b, [ div["tp"] + div["tn"] for div in divs[start_ref_win:start_ref_win+ref_win_size] ])
        b1 = reduce(lambda a,b : a + b, [ div["fp"] + div["fn"] for div in divs[start_ref_win:start_ref_win+ref_win_size] ])
    elif metric == "f1":
        a1 = reduce(lambda a,b : a + b, [ 2 * div["tp"] for div in divs[start_ref_win:start_ref_win+ref_win_size] ])
        b1 = reduce(lambda a,b : a + b, [ div["fp"] + div["fn"] for div in divs[start_ref_win:start_ref_win+ref_win_size] ])

    E1 = a1 / (a1 + b1)
    Var1 = (a1 * b1) / ((a1 + b1) ** 2 * (a1 + b1 + 1))

    # accuracy
    if metric == "accuracy":
        a2 = reduce(lambda a,b : a + b, [ div["tp"] + div["tn"] for div in divs[start_curr_win:start_curr_win+curr_win_size] ])
        b2 = reduce(lambda a,b : a + b, [ div["fp"] + div["fn"] for div in divs[start_curr_win:start_curr_win+curr_win_size] ])
    elif metric == "f1":
        a2 = reduce(lambda a,b : a + b, [ 2 * div["tp"] for div in divs[start_curr_win:start_curr_win+curr_win_size] ])
        b2 = reduce(lambda a,b : a + b, [ div["fp"] + div["fn"] for div in divs[start_curr_win:start_curr_win+curr_win_size] ])

    E2 = a2 / (a2 + b2)
    Var2 = (a2 * b2) / ((a2 + b2) ** 2 * (a2 + b2 + 1))

    delta = E2 - E1
    variance = Var1 + Var2

    t_stat = abs(delta) / np.sqrt(variance + eps) # eps to avoid division by 0

    return delta, t_stat

"""
Actually computes the fraction of points that have been altered, with 
respect to each subgroup's size -- for a specific window.
"""
def _get_altered_in_window(matches_ts_list, altered_frac, window):
    start_win, win_size = window

    # OLD version -- this is wrong because it doesn't take into account the size of each subgroup
    # sizes = np.array([ m.matches.shape[0] for m in matches_ts_list ])
    # return (np.vstack(altered_frac[start_win:start_win+win_size]) * sizes[start_win:start_win+win_size].reshape(-1,1)).sum(axis=0) / sizes[start_win:start_win+win_size].sum()

    start_ndx = start_win
    end_ndx = start_win + win_size
    sizes = np.vstack([ m.matches.sum(axis=0) for m in matches_ts_list ])[start_ndx:end_ndx]
    return (altered_frac[start_ndx:end_ndx] * sizes).sum(axis=0) / sizes.sum(axis=0)


"""
Returns the fraction of altered instances in the "current" window -- for each subgroup.
This information can be used as the ground truth in ranking metrics (i.e. subgroups
should be ranked higher (i.e. more drifted) if they have a higher fraction of altered samples.

Parameters
----------
matches_ts_list : list of Matches (see divexp.py)
    List of matches objects, which includes the subgroups that
    each sample in the batch belongs to; one per batch.
window : tuple
    Window to compute the ground truth for.
altered : list of ndarray
    List of boolean arrays indicating which instances were altered in each batch. If not
    specified, `altered_frac` must be specified and will not be recomputed.
altered_frac : ndarray
    Fraction of altered instances in each batch, for each subgroup. If not specified,
    `altered` must be specified and will be used to compute the fraction of altered samples.

Returns
-------
altered_in_win : ndarray
    Fraction of altered instances in the current window, for each subgroup.
altered_frac : ndarray
    Fraction of altered instances in each window, for each subgroup.
"""
def build_ground_truth(matches_ts_list, window, altered=None, altered_frac=None):

    if altered_frac is None:
        assert altered is not None, "Either altered or altered_frac must be provided"
        # this can be pre-computed
        altered_per_sg = np.vstack([ matches_ts_list[i].matches[altered[i]].sum(axis=0) for i in range(len(matches_ts_list)) ])
        count_per_sg =   np.vstack([ matches_ts_list[i].matches.sum(axis=0) for i in range(len(matches_ts_list)) ])
        count_per_sg[count_per_sg == 0] = 1 # avoid division by 0 (fraction will still be 0)
        altered_frac = altered_per_sg / count_per_sg

    # this changes based on the window being used
    # how many were altered in the last window? (fraction, for each subgroup)
    altered_in_win = _get_altered_in_window(matches_ts_list, altered_frac, window)
    altered_in_win[np.isnan(altered_in_win)] = 0

    return altered_in_win, altered_frac