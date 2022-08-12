#taking the BrokenPowerLaw to explain the dependencies of the input parameters on the SED

# demonstrate dependencies of the values of p1, p2, gamma_max 
#k_e is demonstrated through the agnpy blob features?
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import astropy.units as u
import pkg_resources
import time 
from agnpy.utils.plot import sed_x_label, sed_y_label, load_mpl_rc
# gammapy modules
from gammapy.modeling.models import (
    SpectralModel,
    Parameter,
    SPECTRAL_MODEL_REGISTRY,
    SkyModel,
)
from agnpy.spectra import BrokenPowerLaw
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.emission_regions import Blob
from agnpy.targets import SSDisk, RingDustTorus
from astropy.constants import k_B, m_e, c, G, M_sun
from astropy.table import Table
from astropy.coordinates import Distance
import logging


# constants
mec2 = m_e.to("erg", equivalencies=u.mass_energy())
gamma_size = 400
gamma_to_integrate = np.logspace(0, 7, gamma_size)

class AgnpyEC(SpectralModel):
    """Wrapper of agnpy's non synchrotron, SSC and EC classes. The flux model
    accounts for the Disk and DT's thermal SEDs.
    A broken power law is assumed for the electron spectrum.
    To limit the span of the parameters space, we fit the log10 of the parameters
    whose range is expected to cover several orders of magnitudes (normalisation,
    gammas, size and magnetic field of the blob).
    """

    tag = "EC"
    log10_k_e = Parameter("log10_k_e", -5, min=-20, max=2)
    p1 = Parameter("p1", 2.1, min=1.0, max=5.0)
    p2 = Parameter("p2", 3.1, min=1.0, max=5.0)
    log10_gamma_b = Parameter("log10_gamma_b", 3, min=1, max=6)
    log10_gamma_min = Parameter("log10_gamma_min", 1, min=0, max=4)
    log10_gamma_max = Parameter("log10_gamma_max", 5, min=3, max=8)
    # source general parameters
    z = Parameter("z", 0.1, min=0.01, max=1)
    d_L = Parameter("d_L", "1e27 cm", min=1e25, max=1e33)
    # emission region parameters
    delta_D = Parameter("delta_D", 10, min=1, max=40)
    log10_B = Parameter("log10_B", 0.0, min=-3.0, max=1.0)
    t_var = Parameter("t_var", "600 s", min=10, max=np.pi * 1e7)
    mu_s = Parameter("mu_s", 0.9, min=0.0, max=1.0)
    log10_r = Parameter("log10_r", 17.0, min=16.0, max=20.0)
    # disk parameters
    log10_L_disk = Parameter("log10_L_disk", 45.0, min=42.0, max=48.0)
    log10_M_BH = Parameter("log10_M_BH", 42, min=32, max=45)
    m_dot = Parameter("m_dot", "1e26 g s-1", min=1e24, max=1e30)
    R_in = Parameter("R_in", "1e14 cm", min=1e12, max=1e16)
    R_out = Parameter("R_out", "1e17 cm", min=1e12, max=1e19)
    # DT parameters
    xi_dt = Parameter("xi_dt", 0.6, min=0.0, max=1.0)
    T_dt = Parameter("T_dt", "1e3 K", min=1e2, max=1e4)
    R_dt = Parameter("R_dt", "2.5e18 cm", min=1.0e17, max=1.0e19)

    @staticmethod
    def evaluate(
        energy,
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
    ):
        # conversion
        k_e = 10 ** log10_k_e * u.Unit("cm-3")
        gamma_b = 10 ** log10_gamma_b
        gamma_min = 10 ** log10_gamma_min
        gamma_max = 10 ** log10_gamma_max
        B = 10 ** log10_B * u.G
        R_b = (c * t_var * delta_D / (1 + z)).to("cm")
        r = 10 ** log10_r * u.cm
        L_disk = 10 ** log10_L_disk * u.Unit("erg s-1")
        M_BH = 10 ** log10_M_BH * u.Unit("g")
        eps_dt = 2.7 * (k_B * T_dt / mec2).to_value("")

        nu = energy.to("Hz", equivalencies=u.spectral())
        # non-thermal components
        sed_synch = Synchrotron.evaluate_sed_flux(
            nu,
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
            nu,
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
            nu,
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
        sed_disk = SSDisk.evaluate_multi_T_bb_norm_sed(
            nu, z, L_disk, M_BH, m_dot, R_in, R_out, d_L
        )
        sed_bb_dt = RingDustTorus.evaluate_bb_norm_sed(
            nu, z, xi_dt * L_disk, T_dt, R_dt, d_L
        )
        sed = sed_synch + sed_ssc + sed_ec_dt + sed_disk + sed_bb_dt
        return (sed / energy ** 2).to("1 / (cm2 eV s)")

# IMPORTANT: add the new custom model to the registry of spectral models recognised by gammapy
SPECTRAL_MODEL_REGISTRY.append(AgnpyEC)



# declare a model
agnpy_ec = AgnpyEC()

# initialise parameters
z = 0.361
d_L = Distance(z=z).to("cm")
# - blob
Gamma = 20
delta_D = 25
Beta = np.sqrt(1 - 1 / np.power(Gamma, 2))  # jet relativistic speed
mu_s = (1 - 1 / (Gamma * delta_D)) / Beta  # viewing angle
B = 0.35 * u.G
# - disk
L_disk = 6.7e45 * u.Unit("erg s-1")  # disk luminosity
M_BH = 5.71 * 1e7 * M_sun
eta = 1 / 12
m_dot = (L_disk / (eta * c ** 2)).to("g s-1")
R_g = ((G * M_BH) / c ** 2).to("cm")
R_in = 6 * R_g
R_out = 10000 * R_g
# - DT
xi_dt = 0.6  # fraction of disk luminosity reprocessed by the DT
T_dt = 1e3 * u.K
R_dt = 6.47 * 1e18 * u.cm
# - size and location of the emission region
t_var = 0.5 * u.d
r = 6e17 * u.cm

# instance of the model wrapping angpy functionalities
# - AGN parameters
    # -- distances
agnpy_ec.z.quantity = z
agnpy_ec.z.frozen = True
agnpy_ec.d_L.quantity = d_L.cgs.value
agnpy_ec.d_L.frozen = True
# -- SS disk
agnpy_ec.log10_L_disk.quantity = np.log10(L_disk.to_value("erg s-1"))
agnpy_ec.log10_L_disk.frozen = True
agnpy_ec.log10_M_BH.quantity = np.log10(M_BH.to_value("g"))
agnpy_ec.log10_M_BH.frozen = True
agnpy_ec.m_dot.quantity = m_dot
agnpy_ec.m_dot.frozen = True
agnpy_ec.R_in.quantity = R_in
agnpy_ec.R_in.frozen = True
agnpy_ec.R_out.quantity = R_out
agnpy_ec.R_out.frozen = True
# -- Dust Torus
agnpy_ec.xi_dt.quantity = xi_dt
agnpy_ec.xi_dt.frozen = True
agnpy_ec.T_dt.quantity = T_dt
agnpy_ec.T_dt.frozen = True
agnpy_ec.R_dt.quantity = R_dt
agnpy_ec.R_dt.frozen = True
"""
# - blob parameters
agnpy_ec.delta_D.quantity = delta_D
agnpy_ec.delta_D.frozen = True
agnpy_ec.log10_B.quantity = np.log10(B.to_value("G"))
agnpy_ec.mu_s.quantity = mu_s
agnpy_ec.mu_s.frozen = True
agnpy_ec.t_var.quantity = t_var
agnpy_ec.t_var.frozen = True
agnpy_ec.log10_r.quantity = np.log10(r.to_value("cm"))
agnpy_ec.log10_r.frozen = True

#variable p1 to observe the changes 
for p1 in np.arange(1, 4, 0.5):
    # - EED
    agnpy_ec.log10_k_e.quantity = np.log10(0.05)
    agnpy_ec.p1.quantity = p1
    agnpy_ec.p2.quantity = 3.5
    agnpy_ec.log10_gamma_b.quantity = np.log10(500)
    agnpy_ec.log10_gamma_min.quantity = np.log10(1)
    agnpy_ec.log10_gamma_min.frozen = True
    agnpy_ec.log10_gamma_max.quantity = np.log10(4e4)
    agnpy_ec.log10_gamma_max.frozen = True

    #print(agnpy_ec)

    agnpy_ec.plot(energy_range=[1e-6, 1e15] * u.eV, energy_unit="eV", energy_power=2)
    plt.ylim([1e-13, 1e-8])
    plt.title(r'$\rm {p_1}$ vs the SED')
    plt.legend()

plt.show()

# variable p2 to observe the changes

for p2 in np.arange(1.5, 4, 0.5):
    agnpy_ec.log10_k_e.quantity = np.log10(0.05)
    agnpy_ec.p1.quantity = 1
    agnpy_ec.p2.quantity = p2
    agnpy_ec.log10_gamma_b.quantity = np.log10(350)
    agnpy_ec.log10_gamma_min.quantity = np.log10(1)
    agnpy_ec.log10_gamma_min.frozen = True
    agnpy_ec.log10_gamma_max.quantity = np.log10(1e4)
    agnpy_ec.log10_gamma_max.frozen = True

    #print(agnpy_ec)

    agnpy_ec.plot(energy_range=[1e-6, 1e15] * u.eV, energy_unit="eV", energy_power=2)
    plt.title(r'$\rm {p_2}$ vs the SED')
    plt.ylim([1e-14, 1e-7])
    plt.legend()

plt.show()

# variable gamma_max
for gamma_max in np.arange(1e4, 4e4,0.25e4):
    # - EED
    agnpy_ec.log10_k_e.quantity = np.log10(0.05)
    agnpy_ec.p1.quantity = 1
    agnpy_ec.p2.quantity = 3
    agnpy_ec.log10_gamma_b.quantity = np.log10(350)
    agnpy_ec.log10_gamma_min.quantity = np.log10(1)
    agnpy_ec.log10_gamma_min.frozen = True
    agnpy_ec.log10_gamma_max.quantity = np.log10(gamma_max)
    agnpy_ec.log10_gamma_max.frozen = True
    #print(agnpy_ec)

    agnpy_ec.plot(energy_range=[1e-5, 1e12] * u.eV, energy_unit="eV", energy_power=2)
    plt.ylim([1e-14, 1e-10])
    plt.title(r'$\rm {\gamma_{max}}$ vs the SED')
    plt.legend()

plt.show()

for gamma_b in np.arange(100, 400,25):
    # - EED
    agnpy_ec.log10_k_e.quantity = np.log10(0.05)
    agnpy_ec.p1.quantity = 1
    agnpy_ec.p2.quantity = 3
    agnpy_ec.log10_gamma_b.quantity = np.log10(gamma_b)
    agnpy_ec.log10_gamma_min.quantity = np.log10(1)
    agnpy_ec.log10_gamma_min.frozen = True
    agnpy_ec.log10_gamma_max.quantity = np.log10(4e4)
    agnpy_ec.log10_gamma_max.frozen = True
    #print(agnpy_ec)

    agnpy_ec.plot(energy_range=[1e-5, 1e12] * u.eV, energy_unit="eV", energy_power=2)
    plt.ylim([1e-16, 1e-9])
    plt.title(r'$\rm {\gamma_{b}}$ vs the SED')
    #plt.legend(gamma_b, r'$\rm {\gamma_{b}}$')

plt.show()

#varying log10ke
for log10_k_e in np.arange(0.01, 1,0.05):
    
    # - EED
    agnpy_ec.log10_k_e.quantity = np.log10(log10_k_e)
    agnpy_ec.p1.quantity = 1
    agnpy_ec.p2.quantity = 3
    agnpy_ec.log10_gamma_b.quantity = np.log10(350)
    agnpy_ec.log10_gamma_min.quantity = np.log10(1)
    agnpy_ec.log10_gamma_min.frozen = True
    agnpy_ec.log10_gamma_max.quantity = np.log10(4e4)
    agnpy_ec.log10_gamma_max.frozen = True

    #print(agnpy_ec)

    agnpy_ec.plot(energy_range=[1e-5, 1e12] * u.eV, energy_unit="eV", energy_power=2)
    plt.ylim([1e-16, 1e-7])
    plt.title(r'$\rm {log10_k_e}$ vs the SED')
    

plt.show()
"""
#messing with the blob parameters, comment out the initial set of fixed values of the blob parameter 
# - EED
agnpy_ec.log10_k_e.quantity = np.log10(0.05)
agnpy_ec.p1.quantity = 1
agnpy_ec.p2.quantity = 3
agnpy_ec.log10_gamma_b.quantity = np.log10(350)
agnpy_ec.log10_gamma_min.quantity = np.log10(1)
agnpy_ec.log10_gamma_min.frozen = True
agnpy_ec.log10_gamma_max.quantity = np.log10(4e4)
agnpy_ec.log10_gamma_max.frozen = True
"""
for r in np.arange(4e17,9e17,1e17):
    # - blob parameters
    agnpy_ec.delta_D.quantity = delta_D
    agnpy_ec.delta_D.frozen = True
    agnpy_ec.log10_B.quantity = np.log10(B.to_value("G"))
    agnpy_ec.mu_s.quantity = mu_s
    agnpy_ec.mu_s.frozen = True
    agnpy_ec.t_var.quantity = t_var
    agnpy_ec.t_var.frozen = True
    agnpy_ec.log10_r.quantity = np.log10(r)
    agnpy_ec.log10_r.frozen = True

    agnpy_ec.plot(energy_range=[1e-5, 1e12] * u.eV, energy_unit="eV", energy_power=2)
    plt.ylim([1e-11, 1e-10])
    plt.xlim([1e4,1e12])
    plt.title(r'$\rm {r}$ vs the SED')
    
plt.show()

"""
for delta_D in np.arange(15,100,15):
    # - blob parameters
    agnpy_ec.delta_D.quantity = delta_D
    agnpy_ec.delta_D.frozen = True
    agnpy_ec.log10_B.quantity = np.log10(B.to_value("G"))
    agnpy_ec.mu_s.quantity = mu_s
    agnpy_ec.mu_s.frozen = True
    agnpy_ec.t_var.quantity = t_var
    agnpy_ec.t_var.frozen = True
    agnpy_ec.log10_r.quantity = np.log10(r.to_value("cm"))
    agnpy_ec.log10_r.frozen = True

    agnpy_ec.plot(energy_range=[1e-5, 1e12] * u.eV, energy_unit="eV", energy_power=2)
    plt.ylim([1e-17, 1e-4])
    #plt.xlim([1e-15,1e12])
    plt.title(r'$\rm {\delta_D}$ vs the SED')
    
plt.show()
