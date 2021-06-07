# import numpy, astropy and matplotlib for basic functionalities
import numpy as np
import astropy.units as u
from astropy.constants import k_B, m_e
from astropy.coordinates import Distance
from pathlib import Path
from astropy.table import Table
import matplotlib.pyplot as plt
import pkg_resources
mec2 = m_e.to("erg", equivalencies=u.mass_energy())

# import agnpy classes
import agnpy
from agnpy.emission_regions import Blob
from agnpy.spectra import BrokenPowerLaw
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import RingDustTorus
from agnpy.utils.plot import load_mpl_rc, sed_x_label, sed_y_label, plot_sed
load_mpl_rc()

# import sherpa classes
from sherpa.models import model
from sherpa import data
from sherpa.fit import Fit
from sherpa.stats import Chi2
from sherpa.optmethods import LevMar, NelderMead

class BrokenPowerLawEC_DT(model.RegriddableModel1D):
    """wrapper of agnpy's synchrotron, SSC and EC classes. A broken power-law is assumed for the electron spectrum.
    """

    def __init__(self, name="bpwl_ec"):

        # EED parameters
        self.log10_k_e = model.Parameter(name, "log10_k_e", -1., min=-5., max=1.)
        self.p1 = model.Parameter(name, "p1", 2.1, min=1.0, max=5.0)
        self.delta_p1 = model.Parameter(name, "delta_p1", 1., min=0.0, max=5.0)
        self.log10_gamma_b = model.Parameter(name, "log10_gamma_b", 3., min=1., max=6.)
        self.log10_gamma_min = model.Parameter(name, "log10_gamma_min", 1., min=0., max=4.)
        self.log10_gamma_max = model.Parameter(name, "log10_gamma_max", 5., min=3., max=8.)

        # source general parameters
        self.z = model.Parameter(name, "z", 0.1, min=0.01, max=1)
        self.d_L = model.Parameter(name, "d_L", 1e27, min=1e25, max=1e33)

        #emission region parameters
        self.delta_D = model.Parameter(name, "delta_D", 10, min=1, max=50)
        self.mu_s = model.Parameter(name, "mu_s", 0.9, min=0.0, max=1.0)
        self.log10_B = model.Parameter(name, "log10_B", 0., min=-3., max=1.)
        self.alpha_jet = model.Parameter(name, "alpha_jet", 0.05, min=0.0, max=1.1)
        self.log10_r = model.Parameter(name, "log10_r", 17., min=16., max=20.)

        # DT parameters
        self.log10_L_disk = model.Parameter(name, "log10_L_disk", 45., min=42., max=48.)
        self.xi_dt = model.Parameter(name, "xi_dt", 0.6, min=0.0, max=1.0)
        self.T_dt = model.Parameter(name, "T_dt", 1.e3, min=1.e2, max=1.e4)
        self.R_dt = model.Parameter(name, "R_dt", 2.5e18, min=1.e17, max=1.e19)

        model.RegriddableModel1D.__init__(self, name,
                                          (self.log10_k_e, self.p1, self.delta_p1,
                                           self.log10_gamma_b, self.log10_gamma_min, self.log10_gamma_max,
                                           self.z, self.d_L,
                                           self.delta_D, self.mu_s, self.log10_B, self.alpha_jet, self.log10_r,
                                           self.log10_L_disk, self.xi_dt, self.T_dt, self.R_dt))

    def calc(self, pars, x):
        """evaluate the model calling the agnpy functions"""
        (log10_k_e, p1, delta_p1, log10_gamma_b, log10_gamma_min, log10_gamma_max,
         z, d_L, delta_D, mu_s, log10_B, alpha_jet, log10_r, log10_L_disk, xi_dt, T_dt, R_dt) = pars
        # add units, scale quantities
        x *= u.Hz
        k_e = 10**log10_k_e * u.Unit("cm-3")
        p2 = p1 + delta_p1
        gamma_b = 10**log10_gamma_b
        gamma_min = 10**log10_gamma_min
        gamma_max = 10**log10_gamma_max
        d_L *= u.cm
        B = 10**log10_B * u.G
        L_disk = 10**log10_L_disk * u.Unit("erg s-1")
        R_dt *= u.cm
        T_dt *= u.K
        r = 10**log10_r * u.cm
        R_b = r*alpha_jet
        eps_dt = 2.7 * (k_B * T_dt / mec2).to_value("")

        sed_synch = Synchrotron.evaluate_sed_flux(
            x, z, d_L, delta_D, B, R_b, BrokenPowerLaw, k_e, p1, p2, gamma_b, gamma_min, gamma_max, ssa=True
        )
        sed_ssc = SynchrotronSelfCompton.evaluate_sed_flux(
            x, z, d_L, delta_D, B, R_b, BrokenPowerLaw, k_e, p1, p2, gamma_b, gamma_min, gamma_max
        )
        sed_dt = ExternalCompton.evaluate_sed_flux_dt(
            x, z, d_L, delta_D, mu_s, R_b, L_disk, xi_dt, eps_dt, R_dt, r, BrokenPowerLaw, k_e, p1, p2, gamma_b, gamma_min, gamma_max
        )
        return sed_synch + sed_ssc + sed_dt

        # read the 1D data
sed_path = pkg_resources.resource_filename("agnpy", "data/mwl_seds/PKS1510-089_2015.ecsv")
sed_table = Table.read(sed_path)
x = sed_table["E"]
y = sed_table["nuFnu"]
y_err = sed_table["nuFnu_err_lo"]

# convert to classical SED units
x.convert_unit_to("Hz", equivalencies=u.spectral())
y.convert_unit_to("erg cm-2 s-1")
y_err.convert_unit_to("erg cm-2  s-1")

# plot the data
plt.errorbar(x, y, yerr=y_err, ls="", marker=".", color="k", label="PKS1510-089")
plt.yscale("log")
plt.xscale("log")
plt.xlabel(sed_x_label)
plt.ylabel(sed_y_label)
plt.legend()
plt.show()

# load them in a sherpa data object
sed = data.Data1D("sed", x, y, staterror=y_err)
print(sed)


# global parameters of the blob and the DT

# galaxy distance
z = 0.361
d_L = Distance(z=z).to("cm")

# blob
Gamma = 20
alpha_jet = 1 / Gamma # jet opening angle
delta_D = 20
Beta = np.sqrt(1 - 1 / np.power(Gamma, 2)) # jet relativistic speed
mu_s = (1 - 1 / (Gamma * delta_D)) / Beta # viewing angle

# dust torus
L_disk = 6.7e45 * u.Unit("erg s-1") # disk luminosity
xi_dt = 0.6 # fraction of disk luminosity reprocessed by the DT
R_dt = 6.47e18 * u.cm  # radius of DT
T_dt = 1e3 * u.K

# location of the emission region
r = 6e17 * u.cm

# instance of the model wrapping angpy functionalities
model = BrokenPowerLawEC_DT()

# EED parameters
model.log10_k_e = np.log10(0.1)
model.p1.val = 2.1
model.delta_p1 = 1
model.log10_gamma_b = np.log10(150)
model.log10_gamma_min = np.log10(5)
model.log10_gamma_min.freeze()
model.log10_gamma_max = np.log10(4e4)
model.log10_gamma_max.freeze()

# source general parameters
model.z = z
model.z.freeze()
model.d_L = d_L.cgs.value
model.d_L.freeze()

# emission region parameters
model.delta_D = delta_D
model.delta_D.freeze()
model.mu_s = mu_s
model.mu_s.freeze()
model.alpha_jet  = alpha_jet
model.alpha_jet.freeze()

model.log10_L_disk = np.log10(L_disk.to_value("erg s-1"))
model.log10_L_disk.freeze()
model.xi_dt = 0.6
model.xi_dt.freeze()
model.T_dt = T_dt.to("K").value
model.T_dt.freeze()
model.R_dt = R_dt.to("cm").value
model.R_dt.freeze()

print(model)

# fit using the Levenberg-Marquardt optimiser
fitter = Fit(sed, model, stat=Chi2(), method=LevMar())
min_x = 1e10
max_x = 1e27
sed.notice(min_x, max_x)
print(fitter)

# perform the fit and time it!
results = fitter.fit()
print("-- fit succesful?", results.succeeded)
print(results.format())

# plot the results!
x = np.logspace(np.log10(min_x), np.log10(max_x), 200)
plt.errorbar(sed.x, sed.y, yerr=sed.staterror, marker=".", ls="", color="k")
plot_sed(x, model(x), ls="-", color="crimson")
plt.ylim([1e-16, 1e-8])
plt.show()

