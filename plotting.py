import logging
from pathlib import Path
from ruamel.yaml import YAML
import numpy as np
import astropy.units as u
from astropy.constants import c
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import SSDisk, RingDustTorus
import matplotlib.pyplot as plt
from agnpy.utils.plot import sed_x_label, sed_y_label, load_mpl_rc
from utils import load_sed_data, AgnpyEC


yaml = YAML()
log = logging.getLogger(__name__)
load_mpl_rc()


def plot_sed_all_states():
    """plot all SEDs at ones"""
    log.info("plotting all the states together")
    colors = ["k", "navy", "dodgerblue", "mediumaquamarine","red", "pink"]
    states = ["low", "2012", "2015a", "2015b", "hess_2016", "magic_2016"]
    # define figure
    fig, ax = plt.subplots()
    for state, color in zip(states, colors):
        # load sed
        main_dir = Path(__file__).parent
        sed = load_sed_data(f"{main_dir}/data/PKS1510-089_sed_{state}.ecsv")
        # load dictionary
        pars = yaml.load(Path(f"{main_dir}/results_fixed/{state}/parameters.yaml"))
        # create model
        agnpy_ec = AgnpyEC()
        agnpy_ec.z = pars["z"]
        agnpy_ec.d_L = pars["d_L"]
        # -- SS disk
        agnpy_ec.log10_L_disk = np.log10(pars["L_disk"])
        agnpy_ec.log10_M_BH = np.log10(pars["M_BH"])
        agnpy_ec.m_dot = pars["m_dot"]
        agnpy_ec.R_in = pars["R_in"]
        agnpy_ec.R_out = pars["R_out"]
        # -- Dust Torus
        agnpy_ec.xi_dt = pars["xi_dt"]
        agnpy_ec.T_dt = pars["T_dt"]
        agnpy_ec.R_dt = pars["R_dt"]
        # - blob parameters
        agnpy_ec.delta_D = pars["delta_D"]
        agnpy_ec.log10_B = np.log10(pars["B"])
        agnpy_ec.mu_s = pars["mu_s"]
        agnpy_ec.t_var = pars["t_var"]
        agnpy_ec.log10_r = np.log10(pars["r"])
        # - EED
        agnpy_ec.log10_k_e = np.log10(pars["k_e"])
        agnpy_ec.p1 = pars["p1"]
        agnpy_ec.p2 = pars["p2"]
        agnpy_ec.log10_gamma_b = np.log10(pars["gamma_b"])
        agnpy_ec.log10_gamma_min = np.log10(pars["gamma_min"])
        agnpy_ec.log10_gamma_max = np.log10(pars["gamma_max"])
        nu = np.logspace(10, 30, 300)
        ax.errorbar(
            sed.x,
            sed.y.value,
            yerr=sed.get_error().value,
            color=color,
            marker=".",
            ls="",
        )
        ax.loglog(nu, agnpy_ec(nu), color=color, label=state)

    # final touches to the plot
    ax.set_ylim([1e-14, 1e-8])
    ax.set_xlim([1e8, 1e28])
    ax.set_xlabel(sed_x_label)
    ax.set_ylabel(sed_y_label)
    ax.legend(loc=2,fontsize="xx-small")
    plt.show()
    fig.savefig(f"{main_dir}/results_fixed/sed_all_states.png")


def plot_sed_state(state):
    """plot the SED of a single state displaying all the components"""
    logging.info(f"plot the final model for {state} with the individual components")
    # results dir
    main_dir = Path(__file__).parent
    # load sed
    sed = load_sed_data(f"{main_dir}/data/PKS1510-089_sed_{state}.ecsv")
    # load parameters
    pars = yaml.load(Path(f"{main_dir}/results_fixed/{state}/parameters.yaml"))
    main_dir = Path(__file__).parent
    #pars = yaml.load(Path(f"{main_dir}/results_fixed/{state}/parameters.yaml"))
    k_e = pars["k_e"] * u.Unit("cm-3")
    p1 = pars["p1"]
    p2 = pars["p2"]
    gamma_b = pars["gamma_b"]
    gamma_min = pars["gamma_min"]
    gamma_max = pars["gamma_max"]
    B = pars["B"] * u.G
    r = pars["r"] * u.cm
    delta_D = pars["delta_D"]
    Gamma = 20  # was fixed at 20 before fitting
    R_b = (c * pars["t_var"] * u.s * delta_D / (1 + pars["z"])).to("cm")
    z = pars["z"]
    # blob definition
    parameters = {
        "p1": p1,
        "p2": p2,
        "gamma_b": gamma_b,
        "gamma_min": gamma_min,
        "gamma_max": gamma_max,
    }
    spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
    blob = Blob(
        R_b,
        z,
        delta_D,
        Gamma,
        B,
        k_e,
        spectrum_dict,
        spectrum_norm_type="differential",
        gamma_size=500,
    )

    # Disk and DT definition
    L_disk = pars["L_disk"] * u.Unit("erg s-1")
    M_BH = pars["M_BH"] * u.Unit("g")
    m_dot = pars["m_dot"] * u.Unit("g s-1")
    eta = (L_disk / (m_dot * c ** 2)).to_value("")
    R_in = pars["R_in"] * u.cm
    R_out = pars["R_out"] * u.cm
    disk = SSDisk(M_BH, L_disk, eta, R_in, R_out)
    xi_dt = pars["xi_dt"]
    T_dt = pars["T_dt"] * u.K
    R_dt = pars["R_dt"] * u.cm
    dt = RingDustTorus(L_disk, xi_dt, T_dt, R_dt=R_dt)

    # radiative processes
    synch = Synchrotron(blob, ssa=True)
    ssc = SynchrotronSelfCompton(blob, synch)
    ec_dt = ExternalCompton(blob, dt, r)
    # SEDs
    nu = np.logspace(9, 27, 200) * u.Hz
    synch_sed = synch.sed_flux(nu)
    ssc_sed = ssc.sed_flux(nu)
    ec_dt_sed = ec_dt.sed_flux(nu)
    disk_bb_sed = disk.sed_flux(nu, z)
    dt_bb_sed = dt.sed_flux(nu, z)
    total_sed = synch_sed + ssc_sed + ec_dt_sed + disk_bb_sed + dt_bb_sed

    load_mpl_rc()
    fig, ax = plt.subplots()
    ax.loglog(
        nu / (1 + z), total_sed, ls="-", lw=2.1, color="crimson", label="agnpy, total"
    )
    ax.loglog(
        nu / (1 + z),
        synch_sed,
        ls="--",
        lw=1.3,
        color="goldenrod",
        label="synchrotron",
    )
    ax.loglog(nu / (1 + z), ssc_sed, ls="--", lw=1.3, color="dodgerblue", label="SSC")
    ax.loglog(
        nu / (1 + z),
        ec_dt_sed,
        ls="--",
        lw=1.3,
        color="lightseagreen",
        label="EC on DT",
    )
    ax.loglog(
        nu / (1 + z),
        disk_bb_sed,
        ls="-.",
        lw=1.3,
        color="dimgray",
        label="disk blackbody",
    )
    ax.loglog(
        nu / (1 + z), dt_bb_sed, ls=":", lw=1.3, color="dimgray", label="DT blackbody",
    )
    # systematics error in gray
    ax.errorbar(
        sed.x,
        sed.y.value,
        yerr=sed.get_syserror().value,
        marker=",",
        ls="",
        color="gray",
        label="",
    )
    # statistics error in black
    ax.errorbar(
        sed.x,
        sed.y.value,
        yerr=sed.get_staterror().value,
        marker=".",
        ls="",
        color="k",
        label=f"PKS 1510-089, {state}",
    )
    ax.set_xlabel(sed_x_label)
    ax.set_ylabel(sed_y_label)
    ax.set_xlim([1e9, 1e29])
    ax.set_ylim([10 ** (-14), 10 ** (-7)])
    ax.legend(
        loc="upper center", fontsize=10, ncol=2,
    )
    Path("results_fixed").mkdir(exist_ok=True)
    fig.savefig(f"results_fixed/sed_{state}.png")
