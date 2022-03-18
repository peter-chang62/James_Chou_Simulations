import numpy as np
import matplotlib.pyplot as plt
import simulationHeader as sh
import clipboard_and_style_sheet
import scipy.constants as sc
import pynlo_peter.Fiber_PPLN_NLSE as fpn

clipboard_and_style_sheet.style_sheet()


def get_window(center_wavelength_nm, window_ghz):
    nu = sc.c / (center_wavelength_nm * 1e-9)
    window_hz = window_ghz * 1e9
    nu_up = nu + (window_hz / 2)
    nu_down = nu - (window_hz / 2)

    return (sc.c / nu_up) * 1e6, (sc.c / nu_down) * 1e6


# %%
pulse = sh.get_pulse_data()[1]
coupling_efficiency = 0.8
epp = 17.0 * 0.5 * coupling_efficiency

# %%
# sim = sh.simulate(pulse=pulse, fiber=sh.fiber_adhnlf, length_cm=7.0, epp_nJ=epp, nsteps=200)

# %%
length_pm1550 = 5.5
sim_pm1550 = sh.simulate(pulse=pulse, fiber=sh.fiber_pm1550, length_cm=length_pm1550, epp_nJ=epp, nsteps=200)
sim = sh.simulate(pulse=sim_pm1550.pulse, fiber=sh.fiber_adhnlf, length_cm=5.0, epp_nJ=epp * 10 ** -.1,
                  nsteps=200)

# %%
ll_um, ul_um = get_window(1450, 100)
power = fpn.power_in_window(pulse, sim.AW, ll_um, ul_um, 200)

# %%
fig, ax = plt.subplots(1, 1)
ax.pcolormesh(pulse.wl_um, sim.zs * 1e2 + length_pm1550, sim.AW.__abs__() ** 2, cmap='jet')
ax.axvline(1.450, color='r')
ax.set_xlim(1, 2)
ax.set_xlabel("wavelength ($\mathrm{\mu m}$)")
ax.set_ylabel("distance (cm)")

# %%
fig, ax = plt.subplots(1, 1)
ax.plot(sim.zs * 1e2 + length_pm1550, power * 1e3, '.')
ax.set_xlabel("distance (cm)")
ax.set_ylabel("power (mW)")
