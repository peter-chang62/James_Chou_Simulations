# %% ----- package imports
import matplotlib.pyplot as plt
import numpy as np
import clipboard as cr
import tables
import pynlo
from scipy.integrate import simpson
import copy
import matplotlib


# %% --------------------------------------------------------------------------
# c = 299792458.0
# pJ = 1e-12
# nm = 1e-9
# ps = 1e-12
# W = 1e3

# # nm, pJ/nm, rad
# spectrum = np.genfromtxt("210526_2.0W_spectrum.txt", skip_header=1)
# spectrum[:, 0] = c / (spectrum[:, 0] * nm)
# spectrum[:, 1] *= pJ / nm

# # ps, W, rad
# time = np.genfromtxt("210526_2.0W_pulse.txt", skip_header=1)
# time[:, 0] *= ps
# time[:, 1] *= W

# atom = tables.Float64Atom()
# file = tables.open_file("210526_2.0W_pulse.h5", "w")

# file.create_array(where=file.root, name="v_grid", atom=atom, obj=spectrum[:, 0])
# file.create_array(where=file.root, name="p_v", atom=atom, obj=spectrum[:, 1])
# file.create_array(where=file.root, name="phi_v", atom=atom, obj=spectrum[:, 2])

# file.create_array(where=file.root, name="t_grid", atom=atom, obj=time[:, 0])
# file.create_array(where=file.root, name="p_t", atom=atom, obj=time[:, 1])
# file.create_array(where=file.root, name="phi_t", atom=atom, obj=time[:, 2])

# file.close()

# %% --------------------------------------------------------------------------
# c = 299792458
# nm = 1e-9

# spectrum = np.genfromtxt("experimental_spectrum.txt")
# spectrum[:, 0] = c / (spectrum[:, 0] * nm)

# file = tables.open_file("expeirmental_spectrum.h5", "w")
# atom = tables.Float64Atom()
# file.create_array(where=file.root, name="v_grid", atom=atom, obj=spectrum[:, 0])
# file.create_array(where=file.root, name="p_v", atom=atom, obj=spectrum[:, 1])

# file.close()

# %% --------------------------------------------------------------------------
c = 299792458

n_points = 2**10
min_wl = 800e-9
max_wl = 3e-6
center_wl = 1550e-9
t_fwhm = 50e-15
time_window = 10e-12
# e_p = 1.73e-9 * 10 / 2 * 0.68
e_p = 0.85e-9 * 10 / 2

pulse = pynlo.light.Pulse.Sech(
    n_points,
    c / max_wl,
    c / min_wl,
    c / center_wl,
    e_p,
    t_fwhm,
    min_time_window=time_window,
)
file = tables.open_file("210526_2.0W_pulse.h5", "r")
pulse.import_p_v(file.root.v_grid[:], file.root.p_v[:], phi_v=file.root.phi_v[:])
file.close()

# %% -----
# t_grid = np.arange(-750e-15, 750e-15, 1e-15)
# v_grid, spctgm = pulse.calculate_spectrogram(t_grid)

# fig_spctgm, ax_spctgm = plt.subplots(1, 1)
# ax_spctgm.pcolormesh(t_grid * 1e15, c * 1e6 / v_grid, spctgm.T, cmap="cividis")
# ax_spctgm.set_ylim(1.49, 1.62)
# ax_spctgm.set_xlabel("time (fs)")
# ax_spctgm.set_ylabel("wavelength ($\\mathrm{\\mu m}$)")

# %% --------------------------------------------------------------------------
# fibers
hnlf = pynlo.materials.SilicaFiber()
pm1550 = pynlo.materials.SilicaFiber()
pm1550.load_fiber_from_dict(pynlo.materials.pm1550, axis="slow")
hnlf.load_fiber_from_dict(pynlo.materials.hnlf_5p7_pooja, axis="slow")  # anomalous

model_pm1550 = pm1550.generate_model(pulse)
dz = model_pm1550.estimate_step_size()
result_pm1550 = model_pm1550.simulate(
    5.5e-2,
    dz=dz,
    local_error=1e-6,
    n_records=100,
    plot=None,
)

model_hnlf = hnlf.generate_model(result_pm1550.pulse_out, t_shock=None)
dz = model_hnlf.estimate_step_size()
result_hnlf = model_hnlf.simulate(
    5e-2,
    dz=dz,
    local_error=1e-6,
    n_records=100,
    plot=None,
)

# %% ----- power throughout
(ind_1450_25nm,) = np.logical_and(
    (1450 - 12.5) < pulse.wl_grid * 1e9, pulse.wl_grid * 1e9 < (1450 + 12.5)
).nonzero()

(ind_1450_50nm,) = np.logical_and(
    (1450 - 25) < pulse.wl_grid * 1e9, pulse.wl_grid * 1e9 < (1450 + 25)
).nonzero()

(ind_1770_25nm,) = np.logical_and(
    (1770 - 12.5) < pulse.wl_grid * 1e9, pulse.wl_grid * 1e9 < (1770 + 12.5)
).nonzero()

(ind_1770_50nm,) = np.logical_and(
    (1770 - 25) < pulse.wl_grid * 1e9, pulse.wl_grid * 1e9 < (1770 + 25)
).nonzero()

(ind_1700nm_lp,) = (pulse.wl_grid * 1e9 > 1700).nonzero()

# %% ----- power @ 1450 and 1800 nm
p_v = result_hnlf.p_v

pwr_1450_25nm = (
    simpson(p_v[:, ind_1450_25nm], pulse.v_grid[ind_1450_25nm]) * 200e6 * 1e3
)
pwr_1450_50nm = (
    simpson(p_v[:, ind_1450_50nm], pulse.v_grid[ind_1450_50nm]) * 200e6 * 1e3
)

pwr_1700_lp = simpson(p_v[:, ind_1700nm_lp], pulse.v_grid[ind_1700nm_lp]) * 200e6 * 1e3

pwr_1770_25nm = (
    simpson(p_v[:, ind_1770_25nm], pulse.v_grid[ind_1770_25nm]) * 200e6 * 1e3
)

pwr_1770_50nm = (
    simpson(p_v[:, ind_1770_50nm], pulse.v_grid[ind_1770_50nm]) * 200e6 * 1e3
)

# %% ----- plotting
# fig_2d, ax_2d = result_hnlf.plot("wvl")
fig_2d, ax_2d = plt.subplots(1, 2)
p_v_dB = 10 * np.log10(result_hnlf.p_v)
p_t_dB = 10 * np.log10(result_hnlf.p_t)
p_v_dB -= p_v_dB.max()
p_t_dB -= p_t_dB.max()
ax_2d[0].pcolormesh(
    pulse.wl_grid * 1e6,
    result_hnlf.z * 1e3,
    p_v_dB,
    vmin=-40,
    vmax=0,
    shading="auto",
    cmap="inferno",
)
ax_2d[1].pcolormesh(
    pulse.t_grid * 1e15,
    result_hnlf.z * 1e3,
    p_t_dB,
    vmin=-40,
    vmax=0,
    shading="auto",
    cmap="inferno",
)
ax_2d[1].set_xlim(-1000, 1000)
ax_2d[0].set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_2d[0].set_ylabel("propagation distance (mm)")
ax_2d[1].set_xlabel("time (fs)")
ax_2d[1].set_ylabel("propagation distance (mm)")
ax_2d[0].set_title("frequency domain")
ax_2d[1].set_title("time domain")
fig_2d.tight_layout()

fig_pwr, ax_pwr = plt.subplots(1, 1)
ax_pwr.semilogy(result_hnlf.z * 1e3, pwr_1450_25nm, label="25 nm @ 1450", linewidth=2)
ax_pwr.semilogy(result_hnlf.z * 1e3, pwr_1450_50nm, label="50 nm @ 1450", linewidth=2)
ax_pwr.semilogy(result_hnlf.z * 1e3, pwr_1700_lp, label="1700 LP", linewidth=2)
ax_pwr.semilogy(
    result_hnlf.z * 1e3, pwr_1770_25nm, label="25 nm @ 1770 nm", linewidth=2
)
ax_pwr.semilogy(
    result_hnlf.z * 1e3, pwr_1770_50nm, label="50 nm @ 1770 nm", linewidth=2
)
ax_pwr.set_xlabel("z (mm)")
ax_pwr.set_ylabel("power (mW)")
ax_pwr.legend(loc="best")
ax_pwr.set_ylim(ymin=10**-1)
fig_pwr.tight_layout()

# they would like to know the power spectral density at 1770 nm
p_v_wl = result_hnlf.p_v * model_hnlf.dv_dl  # J / m
p_v_wl *= 200e6 * 1e3 * 1e-9  # power / nm
step = int(np.round(1e-3 / np.diff(result_hnlf.z)[0]))
(ind,) = np.logical_and(result_hnlf.z > 20e-3, result_hnlf.z < 36e-3).nonzero()
subset = p_v_wl[ind][::step]
cmap = matplotlib.colormaps["Reds"]
colors = cmap(np.linspace(0.25, 1, len(subset)))

fig_psd, ax_psd = plt.subplots(1, 1)
(ind,) = np.logical_and(pulse.wl_grid > 1.7e-6, pulse.wl_grid < 1.85e-6).nonzero()
[
    ax_psd.semilogy(pulse.wl_grid[ind] * 1e6, i[ind], color=colors[n])
    for n, i in enumerate(subset)
]
ax_psd.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax_psd.set_ylabel("power spectral density (mW / nm)")
fig_psd.tight_layout()

ax_pwr.axvline(20, color=colors[0], linestyle="--")
ax_pwr.axvline(36, color=colors[-1], linestyle="--")

# %% ----- compare against the figure I emailed to April
# fig, ax = plt.subplots(1, 1)
# ax.pcolormesh(pulse.wl_grid * 1e6, result_hnlf.z, result_hnlf.p_v, cmap="jet")
# ax.set_xlim(1, 2)

# %% ----- compare to experimental spectrum from April
hnlf_2p2 = pynlo.materials.SilicaFiber()
hnlf_2p2.load_fiber_from_dict(pynlo.materials.hnlf_2p2, axis="slow")  # normal
model_hnlf_2p2 = hnlf_2p2.generate_model(pulse, t_shock=None)  # directly into hnlf
dz = model_hnlf_2p2.estimate_step_size()
result_hnlf_2p2 = model_hnlf_2p2.simulate(
    5e-2,
    dz=dz,
    local_error=1e-6,
    n_records=100,
    plot=None,
)

pulse_data = copy.deepcopy(pulse)
file = tables.open_file("expeirmental_spectrum.h5")
pulse_data.import_p_v(file.root.v_grid[:], file.root.p_v[:], phi_v=None)
file.close()

# %% -----
# fig, ax = plt.subplots(1, 1)
# norm = pulse_data.p_v.max()
# save = True
# for n, i in enumerate(tqdm(result_hnlf_2p2.p_v)):
#     ax.clear()
#     ax.semilogy(pulse_data.wl_grid * 1e6, pulse_data.p_v / norm, label="experimental")
#     ax.semilogy(pulse_data.wl_grid * 1e6, i / norm, label="simulated")
#     ax.set_ylim(1e-3, 1.5)
#     ax.set_xlim(1.3892, 1.7096)
#     ax.set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
#     ax.set_ylabel("power spectral density (a.u.)")
#     ax.legend(loc="best")
#     ax.set_title(f"{np.round(result_hnlf_2p2.z[n] * 1e3, 2)} mm")
#     fig.tight_layout()
#     if save:
#         plt.savefig(f"fig/{n}.png", transparent=True, dpi=300)
#     else:
#         plt.pause(0.05)

# %% -----
ind_best = np.sum(abs(result_hnlf_2p2.p_v - pulse_data.p_v), axis=1).argmin()
fig, ax = plt.subplots(2, 1)
norm = pulse_data.p_v.max()
ax[0].semilogy(pulse_data.wl_grid * 1e6, pulse_data.p_v / norm, label="experimental")
ax[0].semilogy(
    pulse_data.wl_grid * 1e6, result_hnlf_2p2.p_v[ind_best] / norm, label="simulated"
)
ax[1].plot(pulse_data.wl_grid * 1e6, pulse_data.p_v / norm, label="experimental")
ax[1].plot(
    pulse_data.wl_grid * 1e6, result_hnlf_2p2.p_v[ind_best] / norm, label="simulated"
)
ax[0].set_ylim(1e-3, 1.5)
ax[0].set_xlim(1.3892, 1.7096)
ax[0].set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax[0].set_ylabel("power spectral density (a.u.)")
ax[0].legend(loc="best")
ax[1].set_xlim(1.3892, 1.7096)
ax[1].set_xlabel("wavelength ($\\mathrm{\\mu m}$)")
ax[1].set_ylabel("power spectral density (a.u.)")
ax[1].legend(loc="best")
fig.suptitle(f"{np.round(result_hnlf_2p2.z[ind_best] * 1e3, 2)} mm")
fig.tight_layout()
