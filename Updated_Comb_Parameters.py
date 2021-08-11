from simulationHeader import *


def get_window(center_wavelength_nm, window_ghz):
    nu = sc.c / (center_wavelength_nm * 1e-9)
    window_hz = window_ghz * 1e9
    nu_up = nu + (window_hz / 2)
    nu_down = nu - (window_hz / 2)

    return (sc.c / nu_up) * 1e6, (sc.c / nu_down) * 1e6


pulse = get_pulse_data(plot=False, frep_MHz=200.0, EPP_nJ=17.0)[1]

sim = simulate(pulse=pulse, fiber=fiber_ndhnlf, length_cm=10., epp_nJ=17,
               nsteps=200)

# suppose you use a 100 GHz pass band around 1.45 micron
ll_um, ul_um = get_window(center_wavelength_nm=1450., window_ghz=100.)

# plot the power evolution around 1.45 micron, with a 100 GHz passband
power = fpn.power_in_window(pulse, sim.AW, ll_um, ul_um, 200) * 1e3
fig, ax = plt.subplots(1, 1)
ax.plot((sim.zs * 1e2), power, '.')
ax.set_xlabel("cm")
ax.set_ylabel("mW")

# plot the spectrum at the "best" point
best_ind = np.argmax(power)
AW_best = sim.AW[best_ind]
ind = (pulse.wl_um > 0).nonzero()
fig, ax = plt.subplots(1, 1)
ax.plot(pulse.wl_um[ind], normalize(abs(AW_best[ind]) ** 2))
ax.set_xlim(1.44, 1.46)

# 2D plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12.18, 4.8])
plot_freq_evolv(sim, ax1, xlims=[1.44, 1.46])
plot_time_evolv(sim, ax2)
