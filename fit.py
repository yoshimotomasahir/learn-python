import math

import numpy as np
import scipy
import scipy.stats
import scipy.optimize
import pandas as pd

def gaussian_func(arr, constant, mean, sigma):
    return constant * np.exp(- (arr - mean) ** 2 / (2 * sigma ** 2))

def fit_gaussian(arr, nbin, min_x, max_x, p0=None, *, verbose=True):

    if len(arr) == 0:
        raise Exception()
    wbin = (max_x - min_x) / nbin

    arr_x = [(i + 0.5) * wbin + min_x for i in range(nbin)]
    arr_y = [0 for _ in range(nbin)]
    arr_set = []

    # Binに詰める
    for x in arr:
        bin = math.floor((x - min_x) / wbin)
        if bin < 0:
            continue
        if bin >= nbin:
            continue
        arr_y[bin] += 1
        arr_set.append(x)

    if len(arr_set) == 0:
        raise Exception()

    if p0 == None:
        p0 = np.array([arr_y[int(np.round((np.mean(arr_set) - min_x) / wbin))], np.mean(arr_set), np.std(arr_set)])
    else:
        if len(p0) != 3:
            raise Exception()

    arr_x2 = []
    arr_y2 = []

    # Entriesが0を除外
    for x,y in zip(arr_x,arr_y):
        if y == 0:continue
        arr_x2.append(x)
        arr_y2.append(y)

    # Y軸のエラー
    arr_yerror2 = [math.sqrt(y) for y in arr_y2]

    popt, pcov = scipy.optimize.curve_fit(gaussian_func, np.array(arr_x2), np.array(arr_y2),
                                          sigma=np.array(arr_yerror2), absolute_sigma=True, p0=p0)

    # 対角行列を取って平方根
    stderr = np.sqrt(np.diag(pcov))  

    arr_fitted_y = gaussian_func(np.array(arr_x2), popt[0], popt[1], popt[2])
    chisq, p = scipy.stats.chisquare(f_exp=arr_y2, f_obs=arr_fitted_y, ddof=2)

    mat = np.vstack((p0, popt,stderr)).T
    df = pd.DataFrame(mat,index=("Constant", "Mean", "Sigma"), columns=("Initial", "Estimate", "+/- Error"))
    if verbose:
        print(df)

    obj = {}
    obj["Chi2"] = chisq
    obj["P-value"] = p
    obj["NDF"] = len(arr_y2) - 3
    obj["Constant"] = [popt[0], stderr[0]]
    obj["Mean"] = [popt[1], stderr[1]]
    obj["Sigma"] = [popt[2], stderr[2]]
    return obj


if __name__ == "__main__":
    arr = []
    with open("gaus_sample.txt") as f:
        lines = f.readlines()
        for line in lines:
            strs = line.split(' ')
            for i in range(int(strs[1])):
                arr.append(float(strs[0]))
            
    fit_gaussian(arr, 100, -5, 5)