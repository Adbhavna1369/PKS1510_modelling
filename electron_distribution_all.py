import astropy.units as u
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc
from agnpy.spectra import BrokenPowerLaw, LogParabola

# set the remaining quantities defining the blob
R_b = 2.38e16 * u.cm
z = 0.361
delta_D = 25
Gamma = 20

# matplotlib adjustments
load_mpl_rc()

# set the spectrum normalisation (total energy in electrons in this case)
spectrum_norm = 1e48 * u.Unit("erg")
# define the spectral function parametrisation through a dictionary
spectrum_dict_2012 = {
    "type": "LogParabola",
    "parameters": {"q": 2.1422e-01 , "gamma_min": 1, "gamma_max": 4.00e+04, "gamma_0" : 2.49e+01 , "p": 2.032},
}
spectrum_dict_low = {
    "type": "BrokenPowerLaw",
    "parameters": {"p2": 3.34, "gamma_min": 1, "gamma_max": 4e4 , "gamma_b" : 88.7 , "p1": 1.4 },
}
spectrum_dict_2015a = {
    "type": "BrokenPowerLaw",
    "parameters": {"p2": 3.28, "gamma_min": 1, "gamma_max": 4.5e4, "gamma_b" : 727 , "p1": 1.53 },
}
spectrum_dict_2015b = {
    "type": "BrokenPowerLaw",
    "parameters": {"p2": 3.16, "gamma_min": 1, "gamma_max": 3e4, "gamma_b" :1.03e3  , "p1": 2 },
}

blob_low = Blob(R_b, z, delta_D, Gamma, 0.538 * u.G, spectrum_norm, spectrum_dict_low)
blob_2012 = Blob(R_b, z, delta_D, Gamma, 3.31e-01 * u.G, spectrum_norm, spectrum_dict_2012)
blob_2015a = Blob(R_b, z, delta_D, Gamma, 0.276* u.G, spectrum_norm, spectrum_dict_2015a)
blob_2015b = Blob(R_b, z, delta_D, Gamma, 0.379* u.G, spectrum_norm, spectrum_dict_2015b)

blob_low.plot_n_e(gamma_power=2)
blob_2012.plot_n_e(gamma_power=2)
blob_2015a.plot_n_e(gamma_power=2)
blob_2015b.plot_n_e(gamma_power=2)
plt.title ('Electron energy distribution(agnpy)')
plt.legend(["low_state", "2012", "2015a", "2015b"],loc="upper right", fontsize ="xx-small")
plt.show()

# set the spectrum normalisation (total energy in electrons in this case) derived from the mcmc analysis
# define the spectral function parametrisation through a dictionary
spectrum_dict_2012_mcmc = {
    "type": "LogParabola",
    "parameters": {"q": 0.237 , "gamma_min": 1, "gamma_max":  4.602e4, "gamma_0" : 33.03, "p": 2.025},
}
spectrum_dict_low_mcmc = {
    "type": "BrokenPowerLaw",
    "parameters": {"p2": 3.296, "gamma_min": 1, "gamma_max": 3.9e4, "gamma_b" : 90.57 , "p1": 1.54 },
}
spectrum_dict_2015a_mcmc = {
    "type": "BrokenPowerLaw",
    "parameters": {"p2": 3.361, "gamma_min": 1, "gamma_max": 4.49e4, "gamma_b" : 639.73 , "p1": 1.457 },
}
spectrum_dict_2015b_mcmc = {
    "type": "BrokenPowerLaw",
    "parameters": {"p2": 3.139, "gamma_min": 1, "gamma_max": 2.991e4, "gamma_b" : 1.238e3 , "p1": 2.042 },
}



blob_low_mcmc = Blob(R_b, z, delta_D, Gamma, 0.294 * u.G, spectrum_norm, spectrum_dict_low_mcmc)
blob_2012_mcmc = Blob(R_b, z, delta_D, Gamma, 0.3140* u.G, spectrum_norm, spectrum_dict_2012_mcmc)
blob_2015a_mcmc = Blob(R_b, z, delta_D, Gamma, 0.277 * u.G, spectrum_norm, spectrum_dict_2015a_mcmc)
blob_2015b_mcmc = Blob(R_b, z, delta_D, Gamma, 0.356 * u.G, spectrum_norm, spectrum_dict_2015b_mcmc)


# plot the electron distribution
blob_low_mcmc.plot_n_e(gamma_power=2)
blob_2012_mcmc.plot_n_e(gamma_power=2)
blob_2015a_mcmc.plot_n_e(gamma_power=2)
blob_2015b_mcmc.plot_n_e(gamma_power=2)

plt.title ('Electron energy distribution(mcmc analysis)')
plt.legend(["low_state", "2012", "2015a", "2015b"],loc="upper right", fontsize ="xx-small")
plt.show()

