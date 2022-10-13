import numpy as np
import matplotlib.pyplot as plt
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import pynlo_peter.fiber_SSFM_sim_header as sh
import scipy.constants as sc
from scipy.interpolate import InterpolatedUnivariateSpline
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()

wl_ll = 1450 - 5
wl_ul = 1450 + 5
fthz_ll = sc.c * 1e-12 / (wl_ul * 1e-9)
fthz_ul = sc.c * 1e-12 / (wl_ll * 1e-9)
df = fthz_ul - fthz_ll
dt = 0.315 / df

# %% ___________________________________________________________________________________________________________________
pulse = fpn.Pulse(
    # T0_ps=.633,
    T0_ps=dt / 1.76,
    center_wavelength_nm=1450,
    time_window_ps=20,
    NPTS=2 ** 10,
    frep_MHz=100,
    EPP_nJ=5
)

# %% ___________________________________________________________________________________________________________________
# set power spectrum to a flat top
# epp = pulse.calc_epp()
# AW = pulse.AW.copy()
# ind_ll = np.argmin(abs(pulse.F_THz - fthz_ll))
# ind_ul = np.argmin(abs(pulse.F_THz - fthz_ul))
# AW[ind_ll:ind_ul] = 1.
# AW[:ind_ll] = 0
# AW[ind_ul:] = 0
# pulse.set_AW(AW)
# pulse.set_epp(epp)

# %% ___________________________________________________________________________________________________________________
# pulse energy
factor = 10
power_mw = 20
epp_nJ = power_mw * 1e-3 * factor

# %% ___________________________________________________________________________________________________________________
sim = sh.simulate(pulse, sh.fiber_pm1550,
                  length_cm=1000,
                  epp_nJ=epp_nJ,
                  nsteps=1000)

# %% ___________________________________________________________________________________________________________________
sh.plot_freq_evolv(sim, xlims=[pulse.wl_um.min(), pulse.wl_um.max()])
sh.plot_time_evolv(sim)

# %% ___________________________________________________________________________________________________________________
ind_end = np.argmin(abs(sim.zs - 30))

fig, ax = plt.subplots(1, 2)
ax[0].plot(pulse.F_THz, sh.normalize(abs(sim.AW[0]) ** 2), label='input')
ax[0].plot(pulse.F_THz, sh.normalize(abs(sim.AW[ind_end]) ** 2), label='output')
ax[0].axvline(fthz_ll, color='r')
ax[0].axvline(fthz_ul, color='r')
ax[0].set_xlabel("$\mathrm{\\nu}$ (THz)")
ax[0].set_xlim(fthz_ll - 5, fthz_ul + 5)

ax[1].plot(pulse.T_ps, sh.normalize(abs(sim.AT[0]) ** 2), label='input')
ax[1].plot(pulse.T_ps, sh.normalize(abs(sim.AT[-1]) ** 2), label='output')
ax[1].set_xlabel("T (ps)")
spl = InterpolatedUnivariateSpline(pulse.T_ps, sh.normalize(abs(sim.AT[0]) ** 2) - .5)
t0 = np.diff(spl.roots()) * 1e3
df = fthz_ul - fthz_ll

[i.legend(loc='best') for i in ax]
