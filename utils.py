# utils for the fitting
import logging
from pathlib import Path
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.constants import k_B, m_e, c, G, M_sun
from agnpy.spectra import BrokenPowerLaw
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import SSDisk, RingDustTorus
from sherpa import data
from sherpa.models import model
from ruamel.yaml import YAML

yaml = YAML()
# get the main logger
log = logging.getLogger(__name__)
# constants
mec2 = m_e.to("erg", equivalencies=u.mass_energy())
gamma_size = 400
gamma_to_integrate = np.logspace(0, 7, gamma_size)


def load_sed_data(sed_path):
    """load a SED in a `~sherpa.data.Data1D` object"""
    log.info(f"loading data from {sed_path}")
    sed_table = Table.read(sed_path)
    x = sed_table["nu"].to("Hz")
    y = sed_table["flux"].to("erg cm-2 s-1")
    y_err_stat = sed_table["flux_err_lo"].to("erg cm-2 s-1")
    # array of systematic errors, will just be summed in quadrature to the statistical error
    # we assume
    # - 30% on VHE gamma-ray instruments
    # - 10% on HE gamma-ray instruments
    # - 10% on X-ray instruments
    # - 5% on lower-energy instruments
    y_err_syst = np.zeros(len(x))
    # define energy ranges
    nu_vhe = (100 * u.GeV).to("Hz", equivalencies=u.spectral())
    nu_he = (0.1 * u.GeV).to("Hz", equivalencies=u.spectral())
    nu_x_ray_max = (300 * u.keV).to("Hz", equivalencies=u.spectral())
    nu_x_ray_min = (0.3 * u.keV).to("Hz", equivalencies=u.spectral())
    vhe_gamma = x >= nu_vhe
    he_gamma = (x >= nu_he) * (x < nu_vhe)
    x_ray = (x >= nu_x_ray_min) * (x < nu_x_ray_max)
    uv_to_radio = x < nu_x_ray_min
    # declare systematics
    y_err_syst[vhe_gamma] = 0.30
    y_err_syst[he_gamma] = 0.10
    y_err_syst[x_ray] = 0.10
    y_err_syst[uv_to_radio] = 0.05
    y_err_syst = y * y_err_syst
    # remove the points with orders of magnitude smaller error, they are upper limits
    UL = y_err_stat < (y * 1e-3)
    x = x[~UL]
    y = y[~UL]
    y_err_stat = y_err_stat[~UL]
    y_err_syst = y_err_syst[~UL]
    # define the data1D object containing it
    return data.Data1D("sed", x, y, staterror=y_err_stat, syserror=y_err_syst)


def write_parameters_to_yaml(agnpy_model, path):
    """write the model parameters in a yaml file"""
    parameters = dict()
    for par in agnpy_model.pars:
        if par.name.startswith("log10_"):
            # store the power of 10 of logarithmic quantities
            # - strip the log10 prefix and elevate to the powe of 10
            name = par.name.split("log10_")[1]
            parameters[name] = float(f"{10**par.val:.3e}")
        else:
            parameters[par.name] = float(f"{par.val:.3e}")
    yaml.dump(parameters, Path(path))


class AgnpyEC(model.RegriddableModel1D):
    """Wrapper of agnpy's non synchrotron, SSC and EC classes. The flux model
    accounts for the Disk and DT's thermal SEDs. 
    A broken power law is assumed for the electron spectrum.
    To limit the span of the parameters space, we fit the log10 of the parameters 
    whose range is expected to cover several orders of magnitudes (normalisation, 
    gammas, size and magnetic field of the blob). 
    """

    def __init__(self, name="ec"):

        # EED parameters
        self.log10_k_e = model.Parameter(name, "log10_k_e", -2.0, min=-20.0, max=10.0)
        self.p1 = model.Parameter(name, "p1", 2.1, min=-2.0, max=5.0)
        self.p2 = model.Parameter(name, "p2", 3.1, min=-2.0, max=5.0)
        self.log10_gamma_b = model.Parameter(name, "log10_gamma_b", 3, min=1, max=6)
        self.log10_gamma_min = model.Parameter(name, "log10_gamma_min", 1, min=0, max=4)
        self.log10_gamma_max = model.Parameter(name, "log10_gamma_max", 5, min=4, max=8)
        # source general parameters
        self.z = model.Parameter(name, "z", 0.1, min=0.01, max=1)
        self.d_L = model.Parameter(name, "d_L", 1e27, min=1e25, max=1e33, units="cm")
        # emission region parameters
        self.delta_D = model.Parameter(name, "delta_D", 10, min=0, max=40)
        self.log10_B = model.Parameter(name, "log10_B", -2, min=-4, max=2)
        self.t_var = model.Parameter(
            name, "t_var", 600, min=10, max=np.pi * 1e7, units="s"
        )
        self.mu_s = model.Parameter(name, "mu_s", 0.9, min=0.0, max=1.0)
        self.log10_r = model.Parameter(name, "log10_r", 17.0, min=16.0, max=20.0)
        # disk parameters
        self.log10_L_disk = model.Parameter(
            name, "log10_L_disk", 45.0, min=42.0, max=48.0
        )
        self.log10_M_BH = model.Parameter(name, "log10_M_BH", 42, min=32, max=45)
        self.m_dot = model.Parameter(
            name, "m_dot", 1e26, min=1e24, max=1e30, units="g s-1"
        )
        self.R_in = model.Parameter(name, "R_in", 1e14, min=1e12, max=1e16, units="cm")
        self.R_out = model.Parameter(
            name, "R_out", 1e17, min=1e12, max=1e19, units="cm"
        )
        # DT parameters
        self.xi_dt = model.Parameter(name, "xi_dt", 0.6, min=0.0, max=1.0)
        self.T_dt = model.Parameter(
            name, "T_dt", 1.0e3, min=1.0e2, max=1.0e4, units="K"
        )
        self.R_dt = model.Parameter(
            name, "R_dt", 2.5e18, min=1.0e17, max=1.0e19, units="cm"
        )

        model.RegriddableModel1D.__init__(
            self,
            name,
            (
                self.log10_k_e,
                self.p1,
                self.p2,
                self.log10_gamma_b,
                self.log10_gamma_min,
                self.log10_gamma_max,
                self.z,
                self.d_L,
                self.delta_D,
                self.log10_B,
                self.t_var,
                self.mu_s,
                self.log10_r,
                self.log10_L_disk,
                self.log10_M_BH,
                self.m_dot,
                self.R_in,
                self.R_out,
                self.xi_dt,
                self.T_dt,
                self.R_dt,
            ),
        )

    def calc(self, pars, x):
        """evaluate the model calling the agnpy functions"""
        (
            log10_k_e,
            p1,
            p2,
            log10_gamma_b,
            log10_gamma_min,
            log10_gamma_max,
            z,
            d_L,
            delta_D,
            log10_B,
            t_var,
            mu_s,
            log10_r,
            log10_L_disk,
            log10_M_BH,
            m_dot,
            R_in,
            R_out,
            xi_dt,
            T_dt,
            R_dt,
        ) = pars
        # add units, scale quantities
        x *= u.Hz
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        B = 10 ** log10_B * u.G
        R_b = c.to_value("cm s-1") * t_var * delta_D / (1 + z) * u.cm
        r = 10 ** log10_r * u.cm
        d_L *= u.cm
        L_disk = 10 ** log10_L_disk * u.Unit("erg s-1")
        M_BH = 10 ** log10_M_BH * u.Unit("g")
        m_dot *= u.Unit("g s-1")
        R_in *= u.cm
        R_out *= u.cm
        R_dt *= u.cm
        T_dt *= u.K
        eps_dt = 2.7 * (k_B * T_dt / mec2).to_value("")

        # non-thermal components
        sed_synch = Synchrotron.evaluate_sed_flux(
            x,
            z,
            d_L,
            delta_D,
            B,
            R_b,
            BrokenPowerLaw,
            k_e,
            p1,
            p2,
            gamma_b,
            gamma_min,
            gamma_max,
            ssa=True,
            gamma=gamma_to_integrate,
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            x,
            z,
            d_L,
            delta_D,
            B,
            R_b,
            BrokenPowerLaw,
            k_e,
            p1,
            p2,
            gamma_b,
            gamma_min,
            gamma_max,
            ssa=True,
            gamma=gamma_to_integrate,
        )
        sed_ec_dt = ExternalCompton.evaluate_sed_flux_dt(
            x,
            z,
            d_L,
            delta_D,
            mu_s,
            R_b,
            L_disk,
            xi_dt,
            eps_dt,
            R_dt,
            r,
            BrokenPowerLaw,
            k_e,
            p1,
            p2,
            gamma_b,
            gamma_min,
            gamma_max,
            gamma=gamma_to_integrate,
        )
        # thermal components
        sed_bb_disk = SSDisk.evaluate_multi_T_bb_norm_sed(
            x, z, L_disk, M_BH, m_dot, R_in, R_out, d_L
        )
        sed_bb_dt = RingDustTorus.evaluate_bb_norm_sed(
            x, z, xi_dt * L_disk, T_dt, R_dt, d_L
        )
        return sed_synch + sed_ssc + sed_ec_dt + sed_bb_disk + sed_bb_dt
