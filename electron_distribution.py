import astropy.units as u
from astropy.coordinates import Distance
from agnpy.emission_regions import Blob
import matplotlib.pyplot as plt
from agnpy.utils.plot import load_mpl_rc


# matplotlib adjustments
load_mpl_rc()

# set the spectrum normalisation (total energy in electrons in this case)
spectrum_norm = 1e48 * u.Unit("erg")
# define the spectral function parametrisation through a dictionary
spectrum_dict_2012 = {
    "type": "BrokenPowerLaw",
    "parameters": {"p2": 3.55, "gamma_min": 1, "gamma_max": 1e5, "gamma_b" : 5.2e2 , "p1": 1 },
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


#blob_2016.plot_n_e(gamma_power=2)
# set the spectrum normalisation (total energy in electrons in this case) derived from the mcmc analysis
# define the spectral function parametrisation through a dictionary
spectrum_dict_2012_mcmc = {
    "type": "BrokenPowerLaw",
    "parameters": {"p2": 3.591, "gamma_min": 1, "gamma_max": 1e5, "gamma_b" : 547.01, "p1": 1.027 },
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
"""
spectrum_dict_2016 = {
    "type": "Broken PowerLaw",
    "parameters": {"p2": 3.5, "gamma_min": 1, "gamma_max": 6e4},
}
"""
# set the remaining quantities defining the blob
R_b = 2.38e16 * u.cm
#B = 0.35 * u.G
z = 0.361
delta_D = 25
Gamma = 20

blob_low_mcmc = Blob(R_b, z, delta_D, Gamma, 0.294 * u.G, spectrum_norm, spectrum_dict_low_mcmc)
blob_2012_mcmc = Blob(R_b, z, delta_D, Gamma, 0.544 * u.G, spectrum_norm, spectrum_dict_2012_mcmc)
blob_2015a_mcmc = Blob(R_b, z, delta_D, Gamma, 0.277 * u.G, spectrum_norm, spectrum_dict_2015a_mcmc)
blob_2015b_mcmc = Blob(R_b, z, delta_D, Gamma, 0.356 * u.G, spectrum_norm, spectrum_dict_2015b_mcmc)
blob_low = Blob(R_b, z, delta_D, Gamma, 0.538 * u.G, spectrum_norm, spectrum_dict_low)
blob_2012 = Blob(R_b, z, delta_D, Gamma, 0.294 * u.G, spectrum_norm, spectrum_dict_2012)
blob_2015a = Blob(R_b, z, delta_D, Gamma, 0.276* u.G, spectrum_norm, spectrum_dict_2015a)
blob_2015b = Blob(R_b, z, delta_D, Gamma, 0.379* u.G, spectrum_norm, spectrum_dict_2015b)

#blob_2016 = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict_2016)
# plot the electron distribution
blob_low_mcmc.plot_n_e(gamma_power=2)
blob_2012_mcmc.plot_n_e(gamma_power=2)
blob_2015a_mcmc.plot_n_e(gamma_power=2)
blob_2015b_mcmc.plot_n_e(gamma_power=2)
#blob_2016.plot_n_e(gamma_power=2)
plt.title ('Electron energy distribution(mcmc analysis)')
plt.legend(["low_state", "2012", "2015a", "2015b"],loc="upper right", fontsize ="xx-small")
plt.show()

blob_low.plot_n_e(gamma_power=2)
blob_2012.plot_n_e(gamma_power=2)
blob_2015a.plot_n_e(gamma_power=2)
blob_2015b.plot_n_e(gamma_power=2)
plt.title ('Electron energy distribution(agnpy)')
plt.legend(["low_state", "2012", "2015a", "2015b"],loc="upper right", fontsize ="xx-small")
plt.show()
