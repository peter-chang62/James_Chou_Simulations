import numpy as np
from simulationHeader import *
import clipboard_and_style_sheet as cr

cr.style_sheet()


def get_wavelength_window(center_wavelength_nm, window_ghz):
    nu = sc.c / (center_wavelength_nm * 1e-9)
    window_hz = window_ghz * 1e9
    nu_up = nu + (window_hz / 2)
    nu_down = nu - (window_hz / 2)

    return (sc.c / nu_up) * 1e6, (sc.c / nu_down) * 1e6


# %% ___________________________________________ load pulse data _______________________________________________________
pulse = get_pulse_data()[1]

# %% __________________________________________ run simulations_________________________________________________________
# pm1550
sim_pm1550 = simulate(pulse=pulse,
                      fiber=fiber_pm1550,
                      length_cm=5.7,
                      epp_nJ=17.3 * 0.7 * 0.5,  # 1.73 or 2.1 Watts, 200 MHz rep-rate, 70 % coupling efficiency
                      nsteps=200)
# adhnlf
sim_hnlf = simulate(pulse=sim_pm1550.pulse,
                    fiber=fiber_adhnlf,
                    length_cm=3.0,
                    epp_nJ=sim_pm1550.pulse.calc_epp() * 1e9 * 10 ** (-1 / 10),
                    nsteps=200)

# %% _________________________________________ plotting ________________________________________________________________
# suppose you use a 100 GHz pass band around 1.45 micron
bw_nm = 10
ll_um, ul_um = (1450 - bw_nm / 2) * 1e-3, (1450 + bw_nm / 2) * 1e-3  # 10 nm band pass centered at 1450

power_pm1550 = fpn.power_in_window(pulse=pulse,
                                   AW=sim_pm1550.AW,
                                   ll_um=ll_um,
                                   ul_um=ul_um,
                                   frep_MHz=200) * 1e3

power_hnlf = fpn.power_in_window(pulse=pulse,
                                 AW=sim_hnlf.AW,
                                 ll_um=ll_um,
                                 ul_um=ul_um,
                                 frep_MHz=200) * 1e3

power_patchcord = np.hstack([power_pm1550, power_hnlf])
sim_patchcord = copy.deepcopy(sim_hnlf)
sim_patchcord.AW = np.vstack([sim_pm1550.AW, sim_hnlf.AW])
sim_patchcord.AT = np.vstack([sim_pm1550.AT, sim_hnlf.AT])
sim_patchcord.zs += sim_pm1550.zs[-1] + np.diff(sim_pm1550.zs)[0]
sim_patchcord.zs = np.hstack([sim_pm1550.zs, sim_patchcord.zs])
ind_len_patchcord = np.argmin(abs(sim_patchcord.zs * 1e2 - 7.9))
mW_thru_bw = power_patchcord[ind_len_patchcord]

# fig, ax = plt.subplots(1, 1)
# ax.plot((sim_hnlf.zs * 1e2), power_hnlf, '.')
# ax.set_xlabel("cm")
# ax.set_ylabel(f"{10}nm bandpass at 1450 nm (mW)")

fig, ax = plt.subplots(1, 1)
ax.plot((sim_patchcord.zs * 1e2), power_patchcord, '.')
ax.axvline(5.7, color='g', linestyle='--', label=f'HNLF splice point')
ax.axvline(7.9, color='r', linestyle='--', label=f'patchcord length ({np.round(mW_thru_bw, 1)} mW)')
ax.set_xlabel("propagation distance (cm)")
ax.set_ylabel(f"{10}nm bandpass at 1450 nm (mW)")
ax.legend(loc='best')

# plot the spectrum at the "best" point
# best_ind = np.argmax(power_hnlf)
# AW_best = sim_hnlf.AW[best_ind]
ind_wl = (pulse.wl_um > 0).nonzero()
fig, ax = plt.subplots(1, 1)
ax.plot(pulse.wl_um[ind_wl], normalize(abs(sim_patchcord.AW[ind_len_patchcord][ind_wl]) ** 2))
ax.set_xlim(1.2, 2)
ax.set_xlabel("wavelength ($\mathrm{\mu m}$)")

# # 2D plots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12.18, 4.8])
# plot_freq_evolv(sim_hnlf, ax1)
# plot_time_evolv(sim_hnlf, ax2)

# 2D plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=np.array([7.64, 4.7 ]))
plot_freq_evolv(sim_patchcord, ax1)
plot_time_evolv(sim_patchcord, ax2)
# ax1.axhline(7.9, color='r', linestyle='--')
