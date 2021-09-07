import astropy.units as u
from agnpy.emission_regions import Blob
import matplotlib.pyplot as plt
from utils import AgnpyEC
from agnpy.utils.plot import load_mpl_rc
import logging
from pathlib import Path
from ruamel.yaml import YAML

#from utils import load_sed_data, AgnpyEC


# matplotlib adjustments
yaml = YAML()
log = logging.getLogger(__name__)
load_mpl_rc()

def ed_all_states():
"""plot all the electron distributions at once"""
log.info("plotting all the electron distributions together")
colors = ["k", "navy", "dodgerblue", "mediumaquamarine"]
states = ["low", "2012", "2015a", "2015b"] #,"hess_2016","magic_2016"

# define figure
fig, ax = plt.subplots()
for state, color in zip(states, colors):
# results dir
    main_dir = Path(__file__).parent
# load dictionary
    pars = yaml.load(Path(f"{main_dir}/results/{state}/parameters.yaml"))

# set the spectrum normalisation (total energy in electrons in this case)
    spectrum_norm = 1e48 * u.Unit("erg")
# define the spectral function parametrisation through a dictionary
    spectrum_dict = {
        "type": "PowerLaw",
        "parameters": {"p": pars["p2", "gamma_min": pars["gamma_min"], "gamma_max": pars["gamma_max"]},
}
# set the remaining quantities defining the blob
    R_b = pars["R_b"]
    B = pars["B"]
    z = pars["z"]
    delta_D = 25
    Gamma = 20
    blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)

# plot the electron distribution
blob.plot_n_e(gamma_power=2)
plt.legend()
plt.show()
ax.legend(
        loc="upper center", fontsize=10, ncol=2,
    )
    Path("results").mkdir(exist_ok=True)
fig.savefig(f"results/electron_distribution_{state}.png")
