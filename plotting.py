import logging
from pathlib import Path
from ruamel.yaml import YAML
import numpy as np
import matplotlib.pyplot as plt
from agnpy.utils.plot import sed_x_label, sed_y_label, load_mpl_rc
from utils import load_sed_data, AgnpyEC


yaml = YAML()
log = logging.getLogger(__name__)
load_mpl_rc()


def plot_sed_all_states():
    """plot all SEDs at ones"""
    log.info("plotting all the states together")
    colors = ["k", "navy", "dodgerblue", "mediumaquamarine"]
    states = ["low", "2012", "2015a", "2015b"]
    # define figure
    fig, ax = plt.subplots()
    for state, color in zip(states, colors):
        # load sed
        main_dir = Path(__file__).parent
        sed = load_sed_data(f"{main_dir}/data/PKS1510-089_sed_{state}.ecsv")
        # load dictionary
        pars = yaml.load(Path(f"{main_dir}/results/{state}/parameters.yaml"))
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
    ax.legend()
    plt.show()
    fig.savefig(f"{main_dir}/results/sed_all_states.png")
