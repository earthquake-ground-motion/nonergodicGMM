#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#

"""
General Class for extracting Ground Motion Intensity Measures (IMs) from a
set of acceleration time series
"""
from multiprocessing import Pool, cpu_count
import numpy as np
from math import pi, sqrt
from scipy.integrate import cumulative_trapezoid
from scipy import constants
import numba
import matplotlib.pyplot as plt
from dynamicgmm.process.konno_ohmachi import KonnoOhmachi
from dynamicgmm.process.sm_utils import (
    get_time_vector, convert_accel_units, nextpow2, get_velocity_displacement)


@numba.njit
def get_sdof_time_series_fast(acceleration, periods, damping, d_t):
    """Implements Nigam and Jennings algoritm to predict the timeseries response of a
    single degree of freedom (SDOF) oscillator in a numbafied manner

    Args:
        acceleration: Acceleration timeseries
        periods: Spectral target periods of SDOF oscillator
        damping: Damping of SDOF oscillator (fraction)
        d_t: Timestep (s)

    Returns:
        x: [Acceleration, Velocity, Displacement] timeseries
        omega: (2 * pi) / T
    """
    # Instantiate the SDOF response array as a 3D array [len(accel), len(periods), 3]
    # where the three dimensions are acceleration, velocity and displacement respectively
    assert (damping >= 0.0) and (damping <= 1.0), "Damping must be in range [0.0, 1.0]"
    num_steps = acceleration.shape[0]
    num_per = periods.shape[0]
    x = np.zeros((num_steps, num_per, 3), dtype=numba.float64)

    omega = (2. * np.pi) / periods
    omega2 = omega ** 2.
    omega3 = omega ** 3.
    omega_d = omega * np.sqrt(1.0 - (damping ** 2.))

    f1 = (2.0 * damping) / (omega3 * d_t)
    f2 = 1.0 / omega2
    f3 = damping * omega
    f4 = 1.0 / omega_d
    f5 = f3 * f4
    f6 = 2.0 * f3
    e = np.exp(-f3 * d_t)
    s = np.sin(omega_d * d_t)
    c = np.cos(omega_d * d_t)
    g1 = e * s
    g2 = e * c
    h1 = (omega_d * g2) - (f3 * g1)
    h2 = (omega_d * g1) + (f3 * g2)
    for k in range(num_steps - 1):
        dug = acceleration[k + 1] - acceleration[k]
        z_1 = f2 * dug
        z_2 = f2 * acceleration[k]
        z_3 = f1 * dug
        z_4 = z_1 / d_t
        if not k:
            b_val = z_2 - z_3
            a_val = (f5 * b_val) + (f4 * z_4)
        else:
            b_val = x[k - 1, :, 2] + z_2 - z_3
            a_val = f4 * x[k - 1, :, 1] + (f5 * b_val + f4 * z_4)
        x[k, :, 2] = (a_val * g1) + (b_val * g2) + z_3 - z_2 - z_1
        x[k, :, 1] = (a_val * h1) - (b_val * h2) - z_4
        x[k, :, 0] = (-f6 * x[k, :, 1]) - (omega2 * x[k, :, 2])
    return x, omega


class AccelerationResponseSpectrum():
    """
    Evaluate the response spectrum using the algorithm of Nigam & Jennings
    (1969)
    In general this is faster than the classical Newmark-Beta method, and
    can provide estimates of the spectra at frequencies higher than that
    of the sampling frequency.
    """

    def __init__(
        self,
        acceleration: np.ndarray,
        time_step: float,
        periods: np.ndarray,
        damping: float = 0.05,
        units: str = "cm/s/s"
    ):
        """
        Setup the response spectrum calculator
        :param numpy.ndarray time_hist:
            Acceleration time history [Time, Acceleration]
        :param numpy.ndarray periods:
            Spectral periods (s) for calculation
        :param float damping:
            Fractional coefficient of damping
        :param str units:
            Units of the acceleration time history {"g", "m/s", "cm/s/s"}

        """
        self.periods = periods
        self.num_per = len(periods)
        self.acceleration = convert_accel_units(acceleration, units)
        self.damping = damping
        self.d_t = time_step
        self.velocity, self.displacement = get_velocity_displacement(
            self.d_t, self.acceleration)
        self.num_steps = len(self.acceleration)
        self.omega = (2. * np.pi) / self.periods
        self.response_spectrum = None

    def __call__(self):
        """
        Define the response spectrum
        """
        x, omega = get_sdof_time_series_fast(self.acceleration, self.periods,
                                             self.damping, self.d_t)
        x_max = np.max(np.fabs(x), axis=0)
        self.response_spectrum = {
            'Period': self.periods,
            'Acceleration': x_max[:, 0],
            'Velocity': x_max[:, 1],
            'Displacement': x_max[:, 2]}
        self.response_spectrum['Pseudo-Velocity'] = omega * \
            self.response_spectrum['Displacement']
        self.response_spectrum['Pseudo-Acceleration'] = (omega ** 2.) * \
            self.response_spectrum['Displacement']
        time_series = {
            'Time-Step': self.d_t,
            'Acceleration': self.acceleration,
            'Velocity': self.velocity,
            'Displacement': self.displacement,
            'PGA': np.max(np.fabs(self.acceleration)),
            'PGV': np.max(np.fabs(self.velocity)),
            'PGD': np.max(np.fabs(self.displacement))}

        return self.response_spectrum, time_series, x[:, :, 0], x[:, :, 1], x[:, :, 2]


def get_peak_measures(time_step, acceleration, get_vel=False, get_disp=False):
    """
    Returns the peak measures from acceleration, velocity and displacement
    time-series
    :param float time_step:
        Time step of acceleration time series in s
    :param numpy.ndarray acceleration:
        Acceleration time series
    :param bool get_vel:
        Choose to return (and therefore calculate) velocity (True) or otherwise
        (false)
    :returns:
        * pga - Peak Ground Acceleration
        * pgv - Peak Ground Velocity
        * pgd - Peak Ground Displacement
        * velocity - Velocity Time Series
        * dispalcement - Displacement Time series
    """
    pga = np.max(np.fabs(acceleration))
    velocity = None
    displacement = None
    # If displacement is not required then do not integrate to get
    # displacement time series
    if get_disp:
        get_vel = True
    if get_vel:
        velocity = time_step * cumulative_trapezoid(acceleration, initial=0.)
        pgv = np.max(np.fabs(velocity))
    else:
        pgv = None
    if get_disp:
        displacement = time_step * cumulative_trapezoid(velocity, initial=0.)
        pgd = np.max(np.fabs(displacement))
    else:
        pgd = None
    return pga, pgv, pgd, velocity, displacement


def get_fourier_spectrum(time_series, time_step):
    """
    Returns the Fourier spectrum of the time series
    :param numpy.ndarray time_series:
        Array of values representing the time series
    :param float time_step:
        Time step of the time series
    :returns:
        Frequency (as numpy array)
        Fourier Amplitude (as numpy array)
    """
    n_val = nextpow2(len(time_series))
    # numpy.fft.fft will zero-pad records whose length is less than the
    # specified nval
    # Get Fourier spectrum
    fspec = np.fft.fft(time_series, n_val)
    # Get frequency axes
    d_f = 1. / (n_val * time_step)
    freq = d_f * np.arange(0., (n_val / 2.0), 1.0)
    return freq, time_step * np.absolute(fspec[:int(n_val / 2.0)])


def get_effective_amplitude_spectrum(x_component, y_component, time_step):
    """
    """
    assert len(x_component) == len(y_component), \
        "Length of two components not the same"
    xfreq, xfas = get_fourier_spectrum(x_component, time_step)
    yfreq, yfas = get_fourier_spectrum(y_component, time_step)
    return np.sqrt(0.5 * (xfas ** 2. + yfas ** 2.)), xfreq


def plot_fourier_spectrum(time_series, time_step, figure_size=(7, 5),
                          filename=None, filetype="png", dpi=300):
    """
    Plots the Fourier spectrum of a time series
    """
    freq, amplitude = get_fourier_spectrum(time_series, time_step)
    plt.figure(figsize=figure_size)
    plt.loglog(freq, amplitude, 'b-')
    plt.xlabel("Frequency (Hz)", fontsize=14)
    plt.ylabel("Fourier Amplitude", fontsize=14)
    _save_image(filename, plt.gcf(), filetype, dpi)
    plt.show()


def get_hvsr(x_component, x_time_step, y_component, y_time_step, vertical,
             vertical_time_step, smoothing_params):
    """
    :param x_component:
        Time series of the x-component of the data
    :param float x_time_step:
        Time-step (in seconds) of the x-component
    :param y_component:
        Time series of the y-component of the data
    :param float y_time_step:
        Time-step (in seconds) of the y-component
    :param vertical:
        Time series of the vertical of the data
    :param float vertical_time_step:
        Time-step (in seconds) of the vertical component
    :param dict smoothing_params:
        Parameters controlling the smoothing of the individual spectra
        Should contain:
        * 'Function' - Name of smoothing method (e.g. KonnoOhmachi)
        * Controlling parameters
    :returns:
        * horizontal-to-vertical spectral ratio
        * frequency
        * maximum H/V
        * Period of Maximum H/V
    """
    smoother = KonnoOhmachi(smoothing_params)
    # Get x-component Fourier spectrum
    xfreq, xspectrum = get_fourier_spectrum(x_component, x_time_step)
    # Smooth spectrum
    xsmooth = smoother.apply_smoothing(xspectrum, xfreq)
    # Get y-component Fourier spectrum
    yfreq, yspectrum = get_fourier_spectrum(y_component, y_time_step)
    # Smooth spectrum
    ysmooth = smoother.apply_smoothing(yspectrum, yfreq)
    # Take geometric mean of x- and y-components for horizontal spectrum
    hor_spec = np.sqrt(xsmooth * ysmooth)
    # Get vertical Fourier spectrum
    vfreq, vspectrum = get_fourier_spectrum(vertical, vertical_time_step)
    # Smooth spectrum
    vsmooth = smoother.apply_smoothing(vspectrum, vfreq)
    # Get HVSR
    hvsr = hor_spec / vsmooth
    max_loc = np.argmax(hvsr)
    return hvsr, xfreq, hvsr[max_loc], 1.0 / xfreq[max_loc]


def get_response_spectrum(acceleration, time_step, periods, damping=0.05,
                          units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the elastic response spectrum of the acceleration time series.
    :param numpy.ndarray acceleration:
        Acceleration time series
    :param float time_step:
        Time step of acceleration time series in s
    :param numpy.ndarray periods:
        List of periods for calculation of the response spectrum
    :param float damping:
        Fractional coefficient of damping
    :param str units:
        Units of the INPUT ground motion records
    :param str method:
        Choice of method for calculation of the response spectrum
        - "Newmark-Beta"
        - "Nigam-Jennings"
    :returns:
        Outputs from :class: smtk.response_spectrum.BaseResponseSpectrum
    """
    response_spec = AccelerationResponseSpectrum(acceleration,
                                                 time_step,
                                                 periods,
                                                 damping,
                                                 units)
    spectrum, time_series, accel, vel, disp = response_spec()
    spectrum["PGA"] = time_series["PGA"]
    spectrum["PGV"] = time_series["PGV"]
    spectrum["PGD"] = time_series["PGD"]
    return spectrum, time_series, accel, vel, disp


def get_response_spectrum_pair(acceleration_x, time_step_x, acceleration_y,
                               time_step_y, periods, damping=0.05,
                               units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the response spectra of a record pair
    :param numpy.ndarray acceleration_x:
        Acceleration time-series of x-component of record
    :param float time_step_x:
        Time step of x-time series (s)
    :param numpy.ndarray acceleration_y:
        Acceleration time-series of y-component of record
    :param float time_step_y:
        Time step of y-time series (s)
    """
    sax = get_response_spectrum(acceleration_x,
                                time_step_x,
                                periods,
                                damping,
                                units,
                                method)[0]
    say = get_response_spectrum(acceleration_y,
                                time_step_y,
                                periods,
                                damping,
                                units,
                                method)[0]
    return sax, say


def geometric_mean_spectrum(sax, say):
    """
    Returns the geometric mean of the response spectrum
    :param dict sax:
        Dictionary of response spectrum outputs from x-component
    :param dict say:
        Dictionary of response spectrum outputs from y-component
    """
    sa_gm = {}
    for key in sax:
        if key == "Period":
            sa_gm[key] = sax[key]
        else:
            sa_gm[key] = np.sqrt(sax[key] * say[key])
    return sa_gm


def arithmetic_mean_spectrum(sax, say):
    """
    Returns the arithmetic mean of the response spectrum
    """
    sa_am = {}
    for key in sax:
        if key == "Period":
            sa_am[key] = sax[key]
        else:
            sa_am[key] = (sax[key] + say[key]) / 2.0
    return sa_am


def envelope_spectrum(sax, say):
    """
    Returns the envelope of the response spectrum
    """
    sa_env = {}
    for key in sax:
        if key == "Period":
            sa_env[key] = sax[key]
        else:
            sa_env[key] = np.max(np.column_stack([sax[key], say[key]]),
                                 axis=1)
    return sa_env


def larger_pga(sax, say):
    """
    Returns the spectral acceleration from the component with the larger PGA
    """
    if sax["PGA"] >= say["PGA"]:
        return sax
    else:
        return say


def rotate_horizontal(series_x, series_y, angle):
    """
    Rotates two time-series according to a specified angle
    :param nunmpy.ndarray series_x:
        Time series of x-component
    :param nunmpy.ndarray series_y:
        Time series of y-component
    :param float angle:
        Angle of rotation (decimal degrees)
    """
    angle = angle * (pi / 180.0)
    rot_hist_x = (np.cos(angle) * series_x) + (np.sin(angle) * series_y)
    rot_hist_y = (-np.sin(angle) * series_x) + (np.cos(angle) * series_y)
    return rot_hist_x, rot_hist_y


def equalise_series(series_x, series_y):
    """
    For two time series from the same record but of different length
    cuts both records down to the length of the shortest record
    N.B. This assumes that the start times and the time-steps of the record
    are the same - if not then this may introduce biases into the record
    :param numpy.ndarray series_x:
         X Time series
    :param numpy.ndarray series_y:
         Y Time series
    """
    n_x = len(series_x)
    n_y = len(series_y)
    if n_x > n_y:
        return series_x[:n_y], series_y
    elif n_y > n_x:
        return series_x, series_y[:n_x]
    else:
        return series_x, series_y


def gmrotdpp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
             percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally-dependent geometric mean
    :param float percentile:
        Percentile of angles (float)
    :returns:
        - Dictionary contaning
        * angles - Array of rotation angles
        * periods - Array of periods
        * GMRotDpp - The rotationally-dependent geometric mean at the specified
                     percentile
        * GeoMeanPerAngle - An array of [Number Angles, Number Periods]
          indicating the Geometric Mean of the record pair when rotated to
          each period
    """
    if (percentile > 100. + 1E-9) or (percentile < 0.):
        raise ValueError("Percentile for GMRotDpp must be between 0. and 100.")
    # Get the time-series corresponding to the SDOF
    sax, _, x_a, _, _ = get_response_spectrum(acceleration_x,
                                              time_step_x,
                                              periods, damping,
                                              units, method)
    say, _, y_a, _, _ = get_response_spectrum(acceleration_y,
                                              time_step_y,
                                              periods, damping,
                                              units, method)
    x_a, y_a = equalise_series(x_a, y_a)
    angles = np.arange(0., 90., 1.)
    max_a_theta = np.zeros([len(angles), len(periods)], dtype=float)
    max_a_theta[0, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                np.max(np.fabs(y_a), axis=0))
    for iloc, theta in enumerate(angles):
        if iloc == 0:
            max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(x_a), axis=0) *
                                           np.max(np.fabs(y_a), axis=0))
        else:
            rot_x, rot_y = rotate_horizontal(x_a, y_a, theta)
            max_a_theta[iloc, :] = np.sqrt(np.max(np.fabs(rot_x), axis=0) *
                                           np.max(np.fabs(rot_y), axis=0))

    gmrotd = np.percentile(max_a_theta, percentile, axis=0)
    return {
        "angles": angles,
        "periods": periods,
        "GMRotDpp": gmrotd,
        "GeoMeanPerAngle": max_a_theta
    }


KEY_LIST = ["PGA", "PGV", "PGD", "Acceleration", "Velocity",
            "Displacement", "Pseudo-Acceleration", "Pseudo-Velocity"]


def gmrotdpp_slow(acceleration_x, time_step_x, acceleration_y, time_step_y,
                  periods, percentile, damping=0.05, units="cm/s/s",
                  method="Nigam-Jennings"):
    """
    Returns the rotationally-dependent geometric mean. This "slow" version
    will rotate the original time-series and calculate the response spectrum
    at each angle. This is a slower process, but it means that GMRotDpp values
    can be calculated for othe time-series parameters (i.e. PGA, PGV and PGD)
    Inputs as for gmrotdpp
    """
    if (percentile > 100. + 1E-9) or (percentile < 0.):
        raise ValueError("Percentile for GMRotDpp must be between 0. and 100.")
    accel_x, accel_y = equalise_series(acceleration_x, acceleration_y)
    angles = np.arange(0., 90., 1.)

    gmrotdpp = {
        "Period": periods,
        "PGA": np.zeros(len(angles), dtype=float),
        "PGV": np.zeros(len(angles), dtype=float),
        "PGD": np.zeros(len(angles), dtype=float),
        "Acceleration": np.zeros([len(angles), len(periods)], dtype=float),
        "Velocity": np.zeros([len(angles), len(periods)], dtype=float),
        "Displacement": np.zeros([len(angles), len(periods)], dtype=float),
        "Pseudo-Acceleration": np.zeros([len(angles), len(periods)],
                                        dtype=float),
        "Pseudo-Velocity": np.zeros([len(angles), len(periods)], dtype=float)}
    # Get the response spectra for each angle
    for iloc, theta in enumerate(angles):
        if np.fabs(theta) < 1E-9:
            rot_x, rot_y = (accel_x, accel_y)
        else:
            rot_x, rot_y = rotate_horizontal(accel_x, accel_y, theta)
        sax, say = get_response_spectrum_pair(rot_x, time_step_x,
                                              rot_y, time_step_y,
                                              periods, damping,
                                              units, method)

        sa_gm = geometric_mean_spectrum(sax, say)
        for key in KEY_LIST:
            if key in ["PGA", "PGV", "PGD"]:
                gmrotdpp[key][iloc] = sa_gm[key]
            else:
                gmrotdpp[key][iloc, :] = sa_gm[key]
    # Get the desired fractile
    for key in KEY_LIST:
        gmrotdpp[key] = np.percentile(gmrotdpp[key], percentile, axis=0)
    return gmrotdpp


def _get_gmrotd_penalty(gmrotd, gmtheta):
    """
    Calculates the penalty function of 4 of Boore, Watson-Lamprey and
    Abrahamson (2006), corresponding to the sum of squares difference between
    the geometric mean of the pair of records and that of the desired GMRotDpp
    :returns:
        "
    """
    n_angles, n_per = np.shape(gmtheta)
    penalty = np.zeros(n_angles, dtype=float)
    coeff = 1. / float(n_per)
    for iloc in range(0, n_angles):
        penalty[iloc] = coeff * np.sum(
            ((gmtheta[iloc, :] / gmrotd) - 1.) ** 2.)

    locn = np.argmin(penalty)
    return locn, penalty


def gmrotipp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
             percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally-independent geometric mean (GMRotIpp)
    """
    acceleration_x, acceleration_y = equalise_series(acceleration_x,
                                                     acceleration_y)
    gmrot = gmrotdpp(acceleration_x, time_step_x, acceleration_y,
                     time_step_y, periods, percentile, damping, units, method)

    min_loc, penalty = _get_gmrotd_penalty(gmrot["GMRotDpp"],
                                           gmrot["GeoMeanPerAngle"])
    target_angle = gmrot["angles"][min_loc]

    rot_hist_x, rot_hist_y = rotate_horizontal(acceleration_x,
                                               acceleration_y,
                                               target_angle)
    sax, say = get_response_spectrum_pair(rot_hist_x, time_step_x,
                                          rot_hist_y, time_step_y,
                                          periods, damping, units, method)

    gmroti = geometric_mean_spectrum(sax, say)
    gmroti["GMRotD{:.2f}".format(percentile)] = gmrot["GMRotDpp"]
    return gmroti


def rotdpp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
           percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally dependent spectrum RotDpp as defined by Boore
    (2010)
    """
    if np.fabs(time_step_x - time_step_y) > 1E-10:
        raise ValueError("Record pair must have the same time-step!")
    acceleration_x, acceleration_y = equalise_series(acceleration_x,
                                                     acceleration_y)
    theta_set = np.arange(0., 180., 1.)
    max_a_theta = np.zeros([len(theta_set), len(periods) + 1])
    max_v_theta = np.zeros_like(max_a_theta)
    max_d_theta = np.zeros_like(max_a_theta)
    for iloc, theta in enumerate(theta_set):
        theta_rad = np.radians(theta)
        arot = acceleration_x * np.cos(theta_rad) +\
            acceleration_y * np.sin(theta_rad)
        saxy = get_response_spectrum(arot, time_step_x, periods, damping,
                                     units, method)[0]
        max_a_theta[iloc, 0] = saxy["PGA"]
        max_a_theta[iloc, 1:] = saxy["Pseudo-Acceleration"]
        max_v_theta[iloc, 0] = saxy["PGV"]
        max_v_theta[iloc, 1:] = saxy["Pseudo-Velocity"]
        max_d_theta[iloc, 0] = saxy["PGD"]
        max_d_theta[iloc, 1:] = saxy["Displacement"]
    rotadpp = np.percentile(max_a_theta, percentile, axis=0)
    rotvdpp = np.percentile(max_v_theta, percentile, axis=0)
    rotddpp = np.percentile(max_d_theta, percentile, axis=0)
    output = {"Pseudo-Acceleration": rotadpp[1:],
              "Pseudo-Velocity": rotvdpp[1:],
              "Displacement": rotddpp[1:],
              "PGA": rotadpp[0],
              "PGV": rotvdpp[0],
              "PGD": rotddpp[0]}
    return output, max_a_theta, max_v_theta, max_d_theta, theta_set


def _get_basic_response_spectrum_for_angle(theta, acceleration_x, acceleration_y,
                                           time_step_x, time_step_y, periods,
                                           damping=0.05, units="cm/s/s"):
    """

    theta: Angle of rotation in radians
    """
    arot = acceleration_x * np.cos(theta) +\
        acceleration_y * np.sin(theta)
    spectrum, time_series, _, _, _ = AccelerationResponseSpectrum(
        arot, time_step_x, periods, damping, units)()
    spectrum["PGA"] = time_series["PGA"]
    spectrum["PGV"] = time_series["PGV"]
    spectrum["PGD"] = time_series["PGD"]
    return spectrum


def rotdpp_parallel(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
                    percentile, damping=0.05, units="cm/s/s", num_proc=None):
    """
    Returns the rotationally dependent spectrum RotDpp as defined by Boore
    (2010) - using parallelisation by theta to speed up calculations
    """
    if np.fabs(time_step_x - time_step_y) > 1E-10:
        raise ValueError("Record pair must have the same time-step!")
    acceleration_x, acceleration_y = equalise_series(acceleration_x,
                                                     acceleration_y)
    theta_set = np.radians(np.arange(0., 180., 1.))
    if num_proc is None:
        num_proc = cpu_count()
    pool = Pool(num_proc)
    A = pool.starmap(
        _get_basic_response_spectrum_for_angle,
        [(theta, acceleration_x, acceleration_y, time_step_x, time_step_y,
          periods, damping, units) for theta in theta_set]
        )
    max_gm_theta = np.zeros([len(theta_set), len(periods) + 1, 3])
    for i, A_i in enumerate(A):
        max_gm_theta[i, :, 0] = np.hstack([A_i["PGA"], A_i["Pseudo-Acceleration"]])
        max_gm_theta[i, :, 1] = np.hstack([A_i["PGV"], A_i["Pseudo-Velocity"]])
        max_gm_theta[i, :, 2] = np.hstack([A_i["PGD"], A_i["Displacement"]])
    pool.close()
    gm_rotdpp = np.percentile(max_gm_theta, percentile, axis=0)
    return {"PGA": gm_rotdpp[0, 0], "PGV": gm_rotdpp[0, 1],
            "PGD": gm_rotdpp[0, 2], "Pseudo-Acceleration": gm_rotdpp[1:, 0],
            "Pseudo-Velocity": gm_rotdpp[1:, 1], "Displacement": gm_rotdpp[1:, 2]}, max_gm_theta


def rotipp(acceleration_x, time_step_x, acceleration_y, time_step_y, periods,
           percentile, damping=0.05, units="cm/s/s", method="Nigam-Jennings"):
    """
    Returns the rotationally independent spectrum RotIpp as defined by
    Boore (2010)
    """
    if np.fabs(time_step_x - time_step_y) > 1E-10:
        raise ValueError("Record pair must have the same time-step!")
    acceleration_x, acceleration_y = equalise_series(acceleration_x,
                                                     acceleration_y)
    target, rota, rotv, rotd, angles = rotdpp(acceleration_x, time_step_x,
                                              acceleration_y, time_step_y,
                                              periods, percentile, damping,
                                              units, method)
    locn, penalty = _get_gmrotd_penalty(
        np.hstack([target["PGA"], target["Pseudo-Acceleration"]]),
        rota)
    target_theta = np.radians(angles[locn])
    arotpp = acceleration_x * np.cos(target_theta) +\
        acceleration_y * np.sin(target_theta)
    spec = get_response_spectrum(arotpp, time_step_x, periods, damping, units,
                                 method)[0]
    spec["GMRot{:2.0f}".format(percentile)] = target
    return spec


ARIAS_FACTOR = pi / (2.0 * (constants.g * 100.))


def get_husid(acceleration, time_step):
    """
    Returns the Husid vector, defined as \int{acceleration ** 2.}
    :param numpy.ndarray acceleration:
        Vector of acceleration values
    :param float time_step:
        Time-step of record (s)
    """
    time_vector = get_time_vector(time_step, len(acceleration))
    husid = np.hstack([0., cumulative_trapezoid(acceleration ** 2., time_vector)])
    return husid, time_vector


def get_arias_intensity(acceleration, time_step, start_level=0., end_level=1.):
    """
    Returns the Arias intensity of the record
    :param float start_level:
        Fraction of the total Arias intensity used as the start time
    :param float end_level:
        Fraction of the total Arias intensity used as the end time
    """
    assert end_level >= start_level
    husid, time_vector = get_husid(acceleration, time_step)
    husid_norm = husid / husid[-1]
    idx = np.where(np.logical_and(husid_norm >= start_level,
                                  husid_norm <= end_level))[0]
    if len(idx) < len(acceleration):
        husid, time_vector = get_husid(acceleration[idx], time_step)
    return ARIAS_FACTOR * husid[-1]


def plot_husid(acceleration, time_step, start_level=0., end_level=1.0,
               figure_size=(7, 5), filename=None, filetype="png", dpi=300):
    """
    Creates a Husid plot for the record
    :param tuple figure_size:
        Size of the output figure (Width, Height)
    :param str filename:
        Name of the file to export
    :param str filetype:
        Type of file for export
    :param int dpi:
        FIgure resolution in dots per inch.
    """
    plt.figure(figsize=figure_size)
    husid, time_vector = get_husid(acceleration, time_step)
    husid_norm = husid / husid[-1]
    idx = np.where(np.logical_and(husid_norm >= start_level,
                                  husid_norm <= end_level))[0]
    plt.plot(time_vector, husid_norm, "b-", linewidth=2.0,
             label="Original Record")
    plt.plot(time_vector[idx], husid_norm[idx], "r-", linewidth=2.0,
             label="%5.3f - %5.3f Arias" % (start_level, end_level))
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Fraction of Arias Intensity", fontsize=14)
    plt.title("Husid Plot")
    plt.legend(loc=4, fontsize=14)
    _save_image(filename, plt.gcf(), filetype, dpi)
    plt.show()


def get_bracketed_duration(acceleration, time_step, threshold):
    """
    Returns the bracketed duration, defined as the time between the first and
    last excursions above a particular level of acceleration
    :param float threshold:
        Threshold acceleration in units of the acceleration time series
    """
    idx = np.where(np.fabs(acceleration) >= threshold)[0]
    if len(idx) == 0:
        # Record does not exced threshold at any point
        return 0.
    else:
        time_vector = get_time_vector(time_step, len(acceleration))
        return time_vector[idx[-1]] - time_vector[idx[0]] + time_step


def get_uniform_duration(acceleration, time_step, threshold):
    """
    Returns the total duration for which the record exceeds the threshold
    """
    idx = np.where(np.fabs(acceleration) >= threshold)[0]
    return time_step * float(len(idx))


def get_significant_duration(acceleration, time_step, start_level=0.,
                             end_level=1.0):
    """
    Returns the significant duration of the record
    """
    assert end_level >= start_level
    husid, time_vector = get_husid(acceleration, time_step)
    idx = np.where(np.logical_and(husid >= (start_level * husid[-1]),
                                  husid <= (end_level * husid[-1])))[0]
    return time_vector[idx[-1]] - time_vector[idx[0]] + time_step


def get_cav(acceleration, time_step, threshold=0.0):
    """
    Returns the cumulative absolute velocity above a given threshold of
    acceleration
    """
    acceleration = np.fabs(acceleration)
    idx = np.where(acceleration >= threshold)
    if len(idx) > 0:
        return np.trapz(acceleration[idx], dx=time_step)
    else:
        return 0.0


def get_cav_standard(acceleration, time_step, units):
    """Returns the standard cumulative absolute velocity (CAV) as defined according
    to Campbell & Bozorgnia (2012).

    CAV_STD = \sum_{i = 1}^N \left( {H\left( {\left| {PGA_i} \right} - 0.025}\ right) \cdot
    \int_{i-1}^i \left| {a \left( t \right)} \right| dt} \right)
    
    where N the number of non-overlapping  1 s time-intervals, |PGA_i| absolute PGA for the
    i^th time interval.
    """
    acceleration_g = convert_accel_units(units, "g")
    time = get_time_vector(time_step, acceleration_g.shape[0])
    second = time.astype(int)
    n_expected = int(1.0 / time_step)
    cav_std = 0.0
    for i in range(0, second[-1]):
        accel_sel = acceleration_g[second == i]
        if len(accel_sel) != n_expected:
            # Not a full 1 second interval - skipping
            continue
        if np.max(np.fabs(accel_sel)) >= 0.025:
            cav_std += np.trapz(accel_sel, dx=time_step)
    return cav_std


def get_arms(acceleration, time_step):
    """
    Returns the root mean square acceleration, defined as
    sqrt{(1 / duration) * \int{acc ^ 2} dt}
    """
    dur = time_step * float(len(acceleration) - 1)
    return np.sqrt((1. / dur) * np.trapz(acceleration ** 2., dx=time_step))


def get_response_spectrum_intensity(spec):
    """
    Returns the response spectrum intensity (Housner intensity), defined
    as the integral of the pseudo-velocity spectrum between the periods of
    0.1 s and 2.5 s
    :param dict spec:
        Response spectrum of the record as output from :class:
        smtk.response_spectrum.BaseResponseSpectrum
    """
    idx = np.where(np.logical_and(spec["Period"] >= 0.1,
                                  spec["Period"] <= 2.5))[0]
    return np.trapz(spec["Pseudo-Velocity"][idx],
                    spec["Period"][idx])


def get_acceleration_spectrum_intensity(spec):
    """
    Returns the acceleration spectrum intensity, defined as the integral
    of the psuedo-acceleration spectrum between the periods of 0.1 and 0.5 s
    """
    idx = np.where(np.logical_and(spec["Period"] >= 0.1,
                                  spec["Period"] <= 0.5))[0]
    return np.trapz(spec["Pseudo-Acceleration"][idx],
                    spec["Period"][idx])


def get_quadratic_intensity(acc_x, acc_y, time_step):
    """
    Returns the quadratic intensity of a pair of records, define as:
    (1. / duration) * \int_0^{duration} a_1(t) a_2(t) dt
    This assumes the time-step of the two records is the same!
    """
    assert len(acc_x) == len(acc_y)
    dur = time_step * float(len(acc_x) - 1)
    return (1. / dur) * np.trapz(acc_x * acc_y, dx=time_step)


def get_principal_axes(time_step, acc_x, acc_y, acc_z=None):
    """
    Returns the principal axes of a set of ground motion records
    """
    # If time-series are not of equal length then equalise
    acc_x, acc_y = equalise_series(acc_x, acc_y)
    if acc_z is not None:
        nhist = 3
        if len(acc_z) > len(acc_x):
            acc_x, acc_z = equalise_series(acc_x, acc_z)
        else:
            acc_x, acc_z = equalise_series(acc_x, acc_z)
            acc_x, acc_y = equalise_series(acc_x, acc_y)
        acc = np.column_stack([acc_x, acc_y, acc_z])
    else:
        nhist = 2
        acc = np.column_stack([acc_x, acc_y])
    # Calculate quadratic intensity matrix
    sigma = np.zeros([nhist, nhist])
    rho = np.zeros([nhist, nhist])
    for iloc in range(0, nhist):
        for jloc in range(0, nhist):
            sigma[iloc, jloc] = get_quadratic_intensity(acc[:, iloc],
                                                        acc[:, jloc],
                                                        time_step)
    # Calculate correlation matrix
    for iloc in range(0, nhist):
        for jloc in range(0, nhist):
            rho[iloc, jloc] = sigma[iloc, jloc] / np.sqrt(sigma[iloc, iloc] *
                                                          sigma[jloc, jloc])
    # Get transformation matrix
    ppal_sigma, phi = np.linalg.eig(sigma)
    # Transform the time-series
    phi = np.matrix(phi)
    acc_trans = phi.T * np.matrix(acc.T)
    acc_1 = acc_trans[0, :].A.flatten()
    acc_2 = acc_trans[1, :].A.flatten()
    if nhist == 3:
        acc_3 = acc_trans[2, :].A.flatten()
    else:
        acc_3 = None

    alpha3z, theta1x = get_rotation_angles(phi, nhist)

    return acc_1, acc_2, acc_3, {"alpha3z": alpha3z, "theta1x": theta1x,
                                 "phi": phi, "sigma": sigma, "rho": rho,
                                 "principal_sigma": np.matrix(ppal_sigma)}


def get_rotation_angles(transf_matrix, nhist):
    """
    Function returns the angle between the third principal axis
    (i.e. the quasi-vertical component and the veritcal axis z
    Function contributed by Cecilia Nieves, UME School, Pavia
    """
    icf = 180.0 / pi
    if nhist == 3:
        alpha3z = icf * np.arccos(transf_matrix[2, 2])
    else:
        alpha3z = 0

    # Angle between axis x and the projection of axis 1 in the xy plane: theta
    if nhist == 3:
        # v_x is a unitary vector in the direction of axis x.
        # v_1 is a unitary vector in the direction of axis 1,
        # then projected onto the xy plane by setting is z coordinate
        # to zero. Theta1x is then the angle between axis x and the
        # projection of axis 1 over the xy plane.

        v_x = np.array([1., 0., 0.])
        inv_transf_matrix = np.linalg.inv(transf_matrix)
        v_1 = inv_transf_matrix[:, 0]
        v_1[2] = 0.0
        theta1x = icf * np.arccos(np.dot(v_x, v_1) / np.linalg.norm(v_1))
        theta1x = float(theta1x)

        # v_y is a unitary vector in the direction of axis y.
        # v_2 is a unitary vector in the direction of axis 2,
        # then projected onto the xy plane by setting is z coordinate
        # to zero.
        v_y = np.array([0., 1., 0.])
        v_2 = inv_transf_matrix[:, 1]
        v_2[2] = 0.0
        theta2y = icf * np.arccos(np.dot(v_y, v_2) / np.linalg.norm(v_2))
        theta2y = float(theta2y)
    else:
        alpha1x = icf * np.arccos(transf_matrix[0, 0])
        theta1x = float(alpha1x)
        theta2y = theta1x

    return alpha3z, theta1x
