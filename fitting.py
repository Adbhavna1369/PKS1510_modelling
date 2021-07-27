# script to fit the different states
import time
import logging
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.coordinates import Distance
from astropy.constants import G, c, M_sun
from agnpy.utils.plot import sed_x_label, sed_y_label
from sherpa.fit import Fit
from sherpa.stats import Chi2
from sherpa.optmethods import LevMar
import matplotlib.pyplot as plt
from utils import load_sed_data, write_parameters_to_yaml, AgnpyEC


# get the main logger
log = logging.getLogger(__name__)


def fit_state(state, k_e, gamma_min, gamma_max, t_var, r):
    """fit one of the states of PKS 1510-089"""
    log.info(f"fitting PKS 1510-089 {state} state")
    # load the data in a sherpa 1D object
    data_dir = Path(__file__).parent / "data"
    sed = load_sed_data(f"{data_dir}/PKS1510-089_sed_{state}.ecsv")
    # declare an instance of the wrapped model
    agnpy_ec = AgnpyEC()
    # global parameters of the blob and the DT
    z = 0.361
    d_L = Distance(z=z).to("cm")
    # blob
    Gamma = 20
    delta_D = 25
    Beta = np.sqrt(1 - 1 / np.power(Gamma, 2))  # jet relativistic speed
    mu_s = (1 - 1 / (Gamma * delta_D)) / Beta  # viewing angle
    B = 0.35 * u.G
    # disk
    L_disk = 6.7e45 * u.Unit("erg s-1")  # disk luminosity
    M_BH = 5.71 * 1e7 * M_sun
    eta = 1 / 12
    m_dot = (L_disk / (eta * c ** 2)).to("g s-1")
    R_g = ((G * M_BH) / c ** 2).to("cm")
    R_in = 6 * R_g
    R_out = 10000 * R_g
    # DT
    xi_dt = 0.6  # fraction of disk luminosity reprocessed by the DT
    T_dt = 1e3 * u.K
    R_dt = 6.47 * 1e18 * u.cm
    # size and location of the emission region
    t_var *= u.d
    r *= u.cm
    # instance of the model wrapping angpy functionalities
    # - AGN parameters
    # -- distances
    agnpy_ec.z = z
    agnpy_ec.z.freeze()
    agnpy_ec.d_L = d_L.cgs.value
    agnpy_ec.d_L.freeze()
    # -- SS disk
    agnpy_ec.log10_L_disk = np.log10(L_disk.to_value("erg s-1"))
    agnpy_ec.log10_L_disk.freeze()
    agnpy_ec.log10_M_BH = np.log10(M_BH.to_value("g"))
    agnpy_ec.log10_M_BH.freeze()
    agnpy_ec.m_dot = m_dot.to_value("g s-1")
    agnpy_ec.m_dot.freeze()
    agnpy_ec.R_in = R_in.to_value("cm")
    agnpy_ec.R_in.freeze()
    agnpy_ec.R_out = R_out.to_value("cm")
    agnpy_ec.R_out.freeze()
    # -- Dust Torus
    agnpy_ec.xi_dt = xi_dt
    agnpy_ec.xi_dt.freeze()
    agnpy_ec.T_dt = T_dt.to_value("K")
    agnpy_ec.T_dt.freeze()
    agnpy_ec.R_dt = R_dt.to_value("cm")
    agnpy_ec.R_dt.freeze()
    # - blob parameters
    agnpy_ec.delta_D = delta_D
    agnpy_ec.delta_D.freeze()
    agnpy_ec.log10_B = np.log10(B.to_value("G"))
    # agnpy_ec.log10_B.freeze()
    agnpy_ec.mu_s = mu_s
    agnpy_ec.mu_s.freeze()
    agnpy_ec.t_var = (t_var).to_value("s")
    agnpy_ec.t_var.freeze()
    agnpy_ec.log10_r = np.log10(r.to_value("cm"))
    # agnpy_ec.log10_r.freeze()
    # - EED
    agnpy_ec.log10_k_e = np.log10(k_e)
    agnpy_ec.p1 = 1.8
    agnpy_ec.p2 = 3.85
    agnpy_ec.log10_gamma_b = np.log10(900)
    agnpy_ec.log10_gamma_min = np.log10(gamma_min)
    agnpy_ec.log10_gamma_min.freeze()
    agnpy_ec.log10_gamma_max = np.log10(gamma_max)
    agnpy_ec.log10_gamma_max.freeze()
    print(agnpy_ec)

    log.info("plotting the initial model (before fit)")
    nu = np.logspace(10, 30, 300)
    plt.errorbar(sed.x, sed.y.value, yerr=sed.get_error().value, marker=".", ls="")
    plt.loglog(nu, agnpy_ec(nu))
    plt.ylim([1e-14, 1e-8])
    plt.xlabel(sed_x_label)
    plt.ylabel(sed_y_label)
    plt.show()

    logging.info("performing the fit")
    # directory to store the checks performed on the fit
    results_dir = f"results/{state}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    # fit using the Levenberg-Marquardt optimiser
    fitter = Fit(sed, agnpy_ec, stat=Chi2(), method=LevMar())
    # choose minimum and maximum energy to fit
    min_x = 1e11 * u.Hz
    max_x = 1e30 * u.Hz
    sed.notice(min_x, max_x)
    # run the fit!
    t_start = time.perf_counter()
    results = fitter.fit()
    t_stop = time.perf_counter()
    delta_t = t_stop - t_start
    logging.info(f"elapsed time: {delta_t:.3f} s")
    print("fit succesful?", results.succeeded)
    print(results.format())

    log.info("plotting the final model (after fit)")
    fig, ax = plt.subplots()
    ax.errorbar(sed.x, sed.y.value, yerr=sed.get_error().value, marker=".", ls="")
    ax.loglog(nu, agnpy_ec(nu))
    ax.set_ylim([1e-14, 1e-8])
    ax.set_xlim([1e8, 1e28])
    ax.set_xlabel(sed_x_label)
    ax.set_ylabel(sed_y_label)
    plt.show()
    fig.savefig(f"{results_dir}/best_fit_sed.png")

    write_parameters_to_yaml(agnpy_ec, f"{results_dir}/parameters.yaml")
