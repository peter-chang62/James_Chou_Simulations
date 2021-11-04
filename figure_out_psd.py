import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
from scipy.interpolate import interp1d
from scipy.integrate import simps


def normalize(vec):
    return vec / np.max(abs(vec))


def df(f0):
    c = sc.c
    dl = 1.0e-9
    return (- c + np.sqrt(c ** 2 + dl ** 2 * f0 ** 2)) / dl


data = np.genfromtxt("after_4.5_cm_ndhnlf_toptica.txt")

wl = data[:, 0] * 1e-9
F = sc.c / wl
psd_uW_nm = data[:, 1]

gridded_psd_uW_nm = interp1d(F, psd_uW_nm, bounds_error=True)

ind = np.logical_and(F > F[-1] + df(F[-1]), F < F[0] - df(F[0])).nonzero()[0]
spectrum = np.zeros(len(psd_uW_nm))
for n in range(ind[0], ind[-1]):
    dF = df(F[n])
    f = np.linspace(F[n] - dF, F[n] + dF, 5000)
    spectrum[n] = simps(gridded_psd_uW_nm(f))
