import numpy as np
import matplotlib.pyplot as plt
import clipboard_and_style_sheet
from simulationHeader import *
from scipy.interpolate import interp1d

clipboard_and_style_sheet.style_sheet()


def get_window(center_wavelength_nm, window_ghz):
    nu = sc.c / (center_wavelength_nm * 1e-9)
    window_hz = window_ghz * 1e9
    nu_up = nu + (window_hz / 2)
    nu_down = nu - (window_hz / 2)

    return (sc.c / nu_up) * 1e6, (sc.c / nu_down) * 1e6


def plot_sim_at_z(sim, z_cm, ax=None, label=None):
    ind_wl = (pulse.wl_um > 0).nonzero()
    ind = np.argmin((sim.zs * 1e2 - z_cm) ** 2)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim(1.4, 1.7)
    AW = sim.AW[ind][ind_wl]
    wl = pulse.wl_um[ind_wl]

    if label is None:
        ax.plot(wl, normalize(abs(AW) ** 2))
    else:
        ax.plot(wl, normalize(abs(AW) ** 2), label=label)


# %%
pulse = get_pulse_data()[1]

# 1.752 W at 200 MHz is 17.52/2 nJ
# They had 45% coupling efficiency
sim = simulate(pulse=pulse, fiber=fiber_ndhnlf, length_cm=6., epp_nJ=17.52 * 0.5 * 0.45, nsteps=400)

# %%
# # suppose you use a 100 GHz pass band around 1.45 micron
# ll_um, ul_um = get_window(center_wavelength_nm=1450., window_ghz=100.)

# # get the power evolution around 1.45 micron, with a 100 GHz passband
# power = fpn.power_in_window(pulse, sim.AW, ll_um, ul_um, 200) * 1e3

# # plot the power evolution
# fig, ax = plt.subplots(1, 1)
# ax.plot((sim.zs * 1e2), power, '.')
# ax.set_xlabel("cm")
# ax.set_ylabel("mW")

# %%
# # get the spectrum at the "best" point
# best_ind = np.argmax(power)
# AW_best = sim.AW[best_ind]

# # plot the spectrum at the "best" point
# ind = (pulse.wl_um > 0).nonzero()
# fig, ax = plt.subplots(1, 1)
# ax.plot(pulse.wl_um[ind], normalize(abs(AW_best[ind]) ** 2))
# ax.set_xlim(1.4, 1.7)

# %%
# # 2D plots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12.18, 4.8])
# plot_freq_evolv(sim, ax1, xlims=[1.3, 1.8])
# plot_time_evolv(sim, ax2)
# plt.savefig("45_percent_coupling_ndhnlf.png")

# %% New Data
after_ndhnlf_exp = np.genfromtxt("after_4.5_cm_ndhnlf_toptica.txt")
wl_um = after_ndhnlf_exp[:, 0] * 1e-3
spectrum = after_ndhnlf_exp[:, 1]

gridded = interp1d(wl_um, spectrum, bounds_error=True)
ind = np.logical_and(pulse.wl_um >= wl_um[0], pulse.wl_um <= wl_um[-1]).nonzero()[0]
gridded_spectrum = gridded(pulse.wl_um[ind])
gridded_spectrum = normalize(gridded_spectrum)

error = np.zeros((len(sim.AW), len(pulse.wl_um[ind])))
for n, aw in enumerate(sim.AW):
    aw2 = abs(aw[ind]) ** 2
    error[n] = (normalize(aw2) - gridded_spectrum) ** 2

error_mean = np.mean(error, axis=1)
best_ind = np.argmin(error_mean)

fig, ax = plt.subplots(2, 2, figsize=np.array([13.78,  6.82]))
[i.set_xlim(*wl_um[[0, -1]]) for i in ax.flatten()]
ax[0, 0].plot(pulse.wl_um[ind], normalize(abs(pulse.AW[ind]) ** 2), label='input spectrum')
ax[0, 1].plot(wl_um, normalize(spectrum), label='data')
ax[0, 1].plot(pulse.wl_um[ind], normalize(abs(sim.AW[best_ind][ind]) ** 2), label='%.2f' % (sim.zs[best_ind] * 100)
                                                                                  + " cm")
ax[1, 0].plot(wl_um, normalize(spectrum), label='data')
plot_sim_at_z(sim, 4.5, ax[1, 0], label='4.5 cm')
ax[1, 1].plot(wl_um, normalize(spectrum), label='data')
plot_sim_at_z(sim, 6, ax[1, 1], label='6 cm')
[i.legend(loc='best') for i in ax.flatten()]
