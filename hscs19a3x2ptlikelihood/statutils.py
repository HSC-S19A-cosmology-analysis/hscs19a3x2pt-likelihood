import numpy as np
from scipy.stats import chi2 as chi2dist
from scipy.integrate import simps
from scipy import stats

def get_pval(chi2, ddof):
    return 1 - stats.chi2.cdf(chi2, ddof)
    
# utils for correlation coeff
def cov2diagonal2d(mat):
    d = np.diag(mat)**0.5
    dd= np.tile(d, (len(d), 1))
    dd = dd*dd.T
    return dd

def cov2correlation_coeff(cov):
    dd = cov2diagonal2d(cov)
    rcc = cov/dd
    return rcc