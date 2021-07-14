import numpy as np
import matplotlib.pyplot as plt
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import clipboard_and_style_sheet

clipboard_and_style_sheet.style_sheet()

normalize = lambda vec: vec / np.max(abs(vec))

pulse_data = np.genfromtxt("210526_2.0W_pulse.txt", skip_header=1)
T_ps = pulse_data[:, 0]
intensity_T = pulse_data[:, 1]
phase_T = pulse_data[:, 2]
amp_T = np.sqrt(intensity_T)
AT = amp_T * np.exp(1j * phase_T)

pulse_from_T = fpn.Pulse(time_window_ps=10, NPTS=2 ** 14, frep_MHz=100., EPP_nJ=5.)
pulse_from_T.set_AT_experiment(T_ps, AT)

spec_data = np.genfromtxt("210526_2.0W_spectrum.txt", skip_header=1)
wl_nm = spec_data[:, 0]
intensity_W = spec_data[:, 1]
phase_W = spec_data[:, 2]
amp_W = np.sqrt(intensity_W)
AW = amp_W * np.exp(1j * phase_W)

pulse_from_W = fpn.Pulse(time_window_ps=10, NPTS=2 ** 14, frep_MHz=100., EPP_nJ=5.)
pulse_from_W.set_AW_experiment(wl_nm * 1e-3, AW)

plt.figure()
plt.plot(pulse_from_T.T_ps, normalize(pulse_from_T.AT.__abs__()**2))
plt.plot(pulse_from_W.T_ps, normalize(pulse_from_W.AT.__abs__()**2))

plt.figure()
plt.plot(pulse_from_W.F_THz, normalize(pulse_from_W.AW.__abs__()**2))
plt.plot(pulse_from_T.F_THz, normalize(pulse_from_T.AW.__abs__()**2))

# TODO It apperas that center is not set correctly...
#  fix that tomorrow
