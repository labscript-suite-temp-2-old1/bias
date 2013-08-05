# pybywire -- functions
# any objects defined or imported in this module are exported through pybywire to labview

from __future__ import division
from numpy import *
from fit_gaussian_2d import *

def compute_OD(atom, flat, dark, cutoff=1, alpha_sigma=1, sigma=1, pixarea=1):
    mask = (flat-dark) < cutoff # to remove
    atom = ma.array(atom-dark, mask=mask, dtype=float32)
    flat = ma.array(flat-dark, mask=mask, dtype=float32)
    ratio = abs(atom/flat)
    od = -log(ratio)
    if alpha_sigma > 0:
        Isat = alpha_sigma
        od += flat/Isat*(1-ratio)
    N = sum(od)*pixarea/sigma
    print 'Computed OD, N =', N
    return {'OD':ma.filled(od,0),'N':N}
