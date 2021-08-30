import numpy as np
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt

dict_list = [
    {
        "file": "/home/pc/PKS1510_modelling/data/PKS1510-089_sed_low.ecsv", #path of the file
        "label": "low state",
        "marker": ".",
        "color": "k",
    },
    {
        "file": "/home/pc/PKS1510_modelling/data/PKS1510-089_sed_2012.ecsv",
        "label": "2012 active state",
        "marker": "+",
        "color": "darkgoldenrod",
    },
    {
        "file": "/home/pc/PKS1510_modelling/data/PKS1510-089_sed_2015a.ecsv",
        "label": "2015 flare a",
        "marker": ".",
        "color": "darkred",
    },
    {
        "file": "/home/pc/PKS1510_modelling/data/PKS1510-089_sed_2015b.ecsv",
        "label": "2015 flare b",
        "marker": ".",
        "color": "crimson",
    },
    {
        "file": "/home/pc/PKS1510_modelling/data/PKS1510-089_sed_hess_2016.ecsv",
        "label": "2016 HESS",
        "marker": "+",
        "color": "teal",
    },
    {
        "file": "/home/pc/PKS1510_modelling/data/PKS1510-089_sed_magic_2016.ecsv",
        "label": "2016 MAGIC",
        "marker": "+",
        "color": "steelblue",
    },
]

fig, ax = plt.subplots()

for dicto in dict_list:
    table = Table.read(dicto["file"])
    ax.errorbar(
        table["nu"],
        table["flux"],
        yerr=[table["flux_err_lo"], table["flux_err_hi"]],
        ls="",
        color=dicto["color"],
        marker=dicto["marker"],
        label=dicto["label"],
    )
ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid()
ax.set_xlabel(r"$\nu \, / \, {\rm Hz}$")
ax.set_ylabel(r"$\nu F_{\nu} \, / \, ({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
plt.show()

fig.savefig("/home/pc/Version 2/plot_states_before_fit/mwl_sed_states.png")
