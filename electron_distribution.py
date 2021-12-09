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
    "type": "PowerLaw",
    "parameters": {"p": 2.598, "gamma_min": 1, "gamma_max": 6e4},
}
spectrum_dict_low = {
    "type": "PowerLaw",
    "parameters": {"p": 3.255, "gamma_min": 1, "gamma_max": 6e4},
}
spectrum_dict_2015a = {
    "type": "PowerLaw",
    "parameters": {"p": 3.007, "gamma_min": 1, "gamma_max": 6e4},
}
spectrum_dict_2015b = {
    "type": "PowerLaw",
    "parameters": {"p": 3.295, "gamma_min": 1, "gamma_max": 6e4},
}
spectrum_dict_2016 = {
    "type": "PowerLaw",
    "parameters": {"p": 3.5, "gamma_min": 1, "gamma_max": 6e4},
}
# set the remaining quantities defining the blob
R_b = 2.75e11 * u.cm
B = 0.35 * u.G
z = 0.361
delta_D = 25
Gamma = 20
blob_low = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict_low)
blob_2012 = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict_2012)
blob_2015a = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict_2015a)
blob_2015b = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict_2015b)
blob_2016 = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict_2016)
# plot the electron distribution
blob_low.plot_n_e(gamma_power=2)
blob_2012.plot_n_e(gamma_power=2)
blob_2015a.plot_n_e(gamma_power=2)
blob_2015b.plot_n_e(gamma_power=2)
blob_2016.plot_n_e(gamma_power=2)

plt.legend(["p = 2.8", "p = 3.255", "p = 2.598", "p = 3.007", "p = 3.295"],loc="upper right", fontsize ="xx-small")
plt.show()
