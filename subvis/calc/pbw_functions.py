# pybywire -- functions
# any objects defined or imported in this module are exported through pybywire to labview

from __future__ import division
from numpy import *
from fit_gaussian_2d import *

def compute_OD(atoms, flat, dark, Icutoff=1, Isat=0, sigma=1, pixel_size=1):
    mask = array(flat-dark, dtype='int16') < Icutoff # ignore pixels without probe light
    atoms = ma.array(atoms-dark, mask=mask)
    flat = ma.array(flat-dark, mask=mask)
    OD = -log(atoms/flat)
    # atoms -= dark   # spurious subtraction, curtailed by mask
    # flat -= dark    # spurious subtraction, curtailed by mask
    # OD = ma.array(-log(atoms/flat), mask=mask)
    if Isat > 0:
        OD += array(flat-atoms, dtype='int16')/Isat
    N = OD.sum()*pixel_size**2/sigma
    print 'Computed OD, N =', N
    return {'OD': OD.filled(nan), 'N':N}
    # return OD