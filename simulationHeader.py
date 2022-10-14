"""
Returning to simulation for James Chou and Yu Liu
"""
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import pynlo_peter.Fiber_PPLN_NLSE as fpn
import clipboard_and_style_sheet
import scipy.constants as sc
import scipy.interpolate as spi

normalize = lambda vec: vec / np.max(abs(vec))


# dB/km to 1/m
def dBkm_to_m(dBkm):
    km = 10 ** (-dBkm / 10)
    return km * 1e-3


# amplitude and phase data from Toptica
def get_pulse_data(plot=False, time_window_ps=10,
                   NPTS=2 ** 14, frep_MHz=100.,
                   EPP_nJ=5., using_notebook=False):
    if using_notebook:
        time_path = "../210526_2.0W_pulse.txt"
        freq_path = "../210526_2.0W_spectrum.txt"
    else:
        time_path = "210526_2.0W_pulse.txt"
        freq_path = "210526_2.0W_spectrum.txt"

    pulse_data = np.genfromtxt(time_path, skip_header=1)
    T_ps = pulse_data[:, 0]
    intensity_T = pulse_data[:, 1]
    phase_T = pulse_data[:, 2]
    amp_T = np.sqrt(intensity_T)
    AT = amp_T * np.exp(1j * phase_T)

    spec_data = np.genfromtxt(freq_path, skip_header=1)
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
    pulse_from_T = fpn.Pulse(time_window_ps=time_window_ps,
                             center_wavelength_nm=center_wavelength_nm,
                             NPTS=NPTS, frep_MHz=frep_MHz,
                             EPP_nJ=EPP_nJ)
    pulse_from_T.set_AT_experiment(T_ps, AT)

    # pulse from setting electric field in frequency domain
    pulse_from_W = fpn.Pulse(time_window_ps=time_window_ps,
                             center_wavelength_nm=center_wavelength_nm,
                             NPTS=NPTS, frep_MHz=frep_MHz,
                             EPP_nJ=EPP_nJ)
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


def simulate(pulse, fiber, length_cm, epp_nJ, nsteps=100):
    pulse: fpn.Pulse
    fiber: fpn.Fiber
    _ = copy.deepcopy(fiber)
    _.length = length_cm * .01
    __ = copy.deepcopy(pulse)
    __.set_epp(epp_nJ * 1.e-9)
    return fpn.FiberFourWaveMixing().propagate(__, _, nsteps)


def get_2d_time_evolv(at2d):
    norm = np.max(abs(at2d) ** 2, axis=1)
    toplot = abs(at2d) ** 2
    toplot = (toplot.T / norm).T
    return toplot


def plot_freq_evolv(sim, ax=None, xlims=None):
    evolv = fpn.get_2d_evolv(sim.AW)

    ind = (sim.pulse.wl_um > 0).nonzero()

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.pcolormesh(sim.pulse.wl_um[ind], (sim.zs * 100.), evolv[:, ind][:, 0, :],
                  cmap='jet',
                  shading='auto')

    if xlims is None:
        ax.set_xlim(1, 2)
    else:
        ax.set_xlim(*xlims)
    ax.set_xlabel("wavelength ($\mathrm{\mu m}$)")
    ax.set_ylabel("propagation distance (cm)")


def plot_time_evolv(sim, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    toplot = get_2d_time_evolv(sim.AT)
    ax.pcolormesh(sim.pulse.T_ps, (sim.zs * 100.), toplot, cmap='jet',
                  shading='auto')
    ax.set_xlim(-1.5, 1.5)
    ax.set_xlabel("time (ps)")
    ax.set_ylabel("propagation distance (cm)")


def video(sim, save=False, figsize=[12.18, 4.8], xlims=None):
    awevolv = fpn.get_2d_evolv(sim.AW)
    atevolv = get_2d_time_evolv(sim.AT)

    ind = (sim.pulse.wl_um > 0).nonzero()

    if save:
        plt.ioff()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if xlims is None:
        xlims = [1.3, 1.9]

    if save:
        for i in range(len(sim.zs)):
            ax1.clear()
            ax2.clear()
            ax2.set_xlim(*xlims)
            ax1.set_xlim(-1, 1)
            ax1.set_xlabel("ps")
            ax2.set_xlabel("$\mathrm{\mu m}$")

            ax1.plot(sim.pulse.T_ps, atevolv[i])
            ax2.plot(sim.pulse.wl_um[ind], awevolv[i][ind])
            fig.suptitle('%.2f' % (sim.zs[i] * 100.))
            plt.savefig("figuresForVideos/" + str(i) + ".png")

            print(str(i + 1) + "/" + str(len(sim.zs)))

        plt.ion()
        return

    for i in range(len(sim.zs)):
        ax1.clear()
        ax2.clear()
        ax2.set_xlim(*xlims)
        ax1.set_xlim(-1, 1)
        ax1.set_xlabel("ps")
        ax2.set_xlabel("$\mathrm{\mu m}$")

        ax1.plot(sim.pulse.T_ps, atevolv[i])
        ax2.plot(sim.pulse.wl_um[ind], awevolv[i][ind])
        fig.suptitle('%.2f' % (sim.zs[i] * 100.))
        plt.pause(.1)


def create_mp4(fps, name):
    command = "ffmpeg -r " + \
              str(fps) + \
              " -f image2 -s 1920x1080 -y -i figuresForVideos/%d.png " \
              "-vcodec libx264 -crf 25  -pix_fmt yuv420p " + \
              name
    os.system(command)


# fiber Parameters:
# OFS AD HNLF parameters
# adhnlf = {
#     "D": 5.4,
#     "Dprime": 0.028,
#     "gamma": 10.9,
#     "Alpha": 0.74,
# }

# Pooja said that these matched her experimental results better
adhnlf = {
    "D": 4.88,
    "Dprime": 0.0228,
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
