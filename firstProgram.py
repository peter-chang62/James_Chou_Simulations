import numpy as np
import matplotlib.pyplot as plt
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import clipboard_and_style_sheet
import scipy.constants as sc
import scipy.interpolate as spi

clipboard_and_style_sheet.style_sheet()

normalize = lambda vec: vec / np.max(abs(vec))


def dBkm_to_m(dBkm):
    km = 10 ** (-dBkm / 10)
    return km * 1e-3


def get_pulse_data(plot=False):
    pulse_data = np.genfromtxt("210526_2.0W_pulse.txt", skip_header=1)
    T_ps = pulse_data[:, 0]
    intensity_T = pulse_data[:, 1]
    phase_T = pulse_data[:, 2]
    amp_T = np.sqrt(intensity_T)
    AT = amp_T * np.exp(1j * phase_T)

    spec_data = np.genfromtxt("210526_2.0W_spectrum.txt", skip_header=1)
    wl_nm = spec_data[:, 0]
    intensity_W = spec_data[:, 1]
    phase_W = spec_data[:, 2]
    amp_W = np.sqrt(intensity_W)
    AW = amp_W * np.exp(1j * phase_W)

    # if setting the electric field in the time domain, it's important you
    # have the center frequency defined correctly!
    F_mks = sc.c / (wl_nm * 1e-9)
    func = spi.interp1d(np.linspace(0, 1, len(F_mks)), F_mks)
    center_wavelength_nm = sc.c / func(.5) * 1e9

    # pulse from setting electric field in time domain
    pulse_from_T = fpn.Pulse(time_window_ps=10,
                             center_wavelength_nm=center_wavelength_nm,
                             NPTS=2 ** 14, frep_MHz=100.,
                             EPP_nJ=5.)
    pulse_from_T.set_AT_experiment(T_ps, AT)

    # pulse from setting electric field in frequency domain
    pulse_from_W = fpn.Pulse(time_window_ps=10,
                             center_wavelength_nm=center_wavelength_nm,
                             NPTS=2 ** 14, frep_MHz=100.,
                             EPP_nJ=5.)
    pulse_from_W.set_AW_experiment(wl_nm * 1e-3, AW)

    if plot:
        """verify things were done correctly"""

        # plotting the amplitude
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(pulse_from_T.T_ps, normalize(pulse_from_T.AT.__abs__() ** 2))
        ax1.plot(pulse_from_W.T_ps, normalize(pulse_from_W.AT.__abs__() ** 2))

        ax2.plot(pulse_from_W.F_THz, normalize(pulse_from_W.AW.__abs__() ** 2))
        ax2.plot(pulse_from_T.F_THz, normalize(pulse_from_T.AW.__abs__() ** 2))
        ax2.set_xlim(183, 202)

        # plotting the real
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(pulse_from_T.T_ps, normalize(pulse_from_T.AT.real))
        ax1.plot(pulse_from_W.T_ps, normalize(pulse_from_W.AT.real))

        ax2.plot(pulse_from_W.F_THz, normalize(pulse_from_W.AW.real))
        ax2.plot(pulse_from_T.F_THz, normalize(pulse_from_T.AW.real))
        ax2.set_xlim(183, 202)

        # plotting the imaginary
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.plot(pulse_from_T.T_ps, normalize(pulse_from_T.AT.imag))
        ax1.plot(pulse_from_W.T_ps, normalize(pulse_from_W.AT.imag))

        ax2.plot(pulse_from_W.F_THz, normalize(pulse_from_W.AW.imag))
        ax2.plot(pulse_from_T.F_THz, normalize(pulse_from_T.AW.imag))
        ax2.set_xlim(183, 202)

        """Looks good!"""

    return pulse_from_T, pulse_from_W


pulse_from_T, pulse_from_W = get_pulse_data()

"""Now let's simulate! The goal is to get an estimate of how much fiber we 
need. I think we ideally want to splice HNLF to a PM-1550 pigtail, cut back 
on the HNLF, and then connectorize. Before that, I want to get an estimate of 
how much broadening we can expect inside PM-1550. At this pulse energy, 
there should definitely be nonlinearity in PM-1550. """

# OFS AD HNLF parameters
adhnlf = {
    "D": 5.4,
    "Dprime": 0.028,
    "gamma": 10.9,
    "Alpha": 0.74,
}

# OFS ND HNLF parameters
ndhnlf = {
    "D": -2.6,
    "Dprime": 0.026,
    "gamma": 10.5,
    "Alpha": 0.8,
}

pm1550 = {
    "D": 18,
    "Dprime": 0.0612,
    "gamma": 1.,
    "Alpha": 0.18
}

fiber_adhnlf = fpn.Fiber()
fiber_adhnlf.generate_fiber(.2,
                            1550.,
                            [adhnlf["D"], adhnlf["Dprime"]],
                            adhnlf["gamma"] * 1e-3,
                            gain=dBkm_to_m(adhnlf["Alpha"]),
                            dispersion_format="D")

fiber_ndhnlf = fpn.Fiber()
fiber_ndhnlf.generate_fiber(.2,
                            1550.,
                            [ndhnlf["D"], ndhnlf["Dprime"]],
                            ndhnlf["gamma"] * 1e-3,
                            gain=dBkm_to_m(ndhnlf["Alpha"]),
                            dispersion_format="D")

fiber_pm1550 = fpn.Fiber()
fiber_pm1550.generate_fiber(.2,
                            1550.,
                            [pm1550["D"], pm1550["Dprime"]],
                            pm1550["gamma"] * 1e-3,
                            gain=dBkm_to_m(pm1550["Alpha"]),
                            dispersion_format="D")

# go through pm1550 first
ssfm = fpn.FiberFourWaveMixing()
pulse_from_W.set_epp(2.e-9)
fiber_pm1550.length = 0.2
res = ssfm.propagate(pulse_from_W, fiber_pm1550, 100)

# then go through some hnlf
pulse_out = res.pulse
fiber2 = fiber_adhnlf
fiber2.length = .1
res2 = ssfm.propagate(pulse_out, fiber2, 100)

# plot the evolution in pm1550
evolv = fpn.get_2d_evolv(res.AW)
plt.figure()
plt.pcolormesh(pulse_from_W.wl_nm * 1e-3, res.zs, evolv, shading='auto')
plt.xlim(1, 2)

# plot the evolution in hnlf
evolv = fpn.get_2d_evolv(res2.AW)
plt.figure()
plt.pcolormesh(pulse_from_W.wl_nm * 1e-3, res2.zs, evolv, shading='auto')
plt.xlim(1, 2)
