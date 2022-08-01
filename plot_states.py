import numpy as np
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt

dict_list = [
    {
        "file": "/home/pc/PKS1510_modelling/data/PKS1510-089_sed_low.ecsv", #path of the file
        "label": "Low State",
        "marker": ".",
        "color": "k",
    },
    {
        "file": "/home/pc/PKS1510_modelling/data/PKS1510-089_sed_2012.ecsv",
        "label": "AS 2012",
        "marker": "+",
        "color": "darkgoldenrod",
    },
    {
        "file": "/home/pc/PKS1510_modelling/data/PKS1510-089_sed_2015a.ecsv",
        "label": "Flare 2015a",
        "marker": ".",
        "color": "darkred",
    },
    {
        "file": "/home/pc/PKS1510_modelling/data/PKS1510-089_sed_2015b.ecsv",
        "label": "Flare 2015b",
        "marker": ".",
        "color": "crimson",
    }
    
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
#ax.legend()
ax.set_xscale("log")
ax.set_yscale("log")
ax.grid()
ax.set_xlabel(r"$\nu \, / \, {\rm Hz}$")
ax.set_ylabel(r"$\nu F_{\nu} \, / \, ({\rm erg}\,{\rm cm}^{-2}\,{\rm s}^{-1})$")
plt.show()

#fig.savefig("/home/pc/PKS1510_modelling-Changes-SED/mwl_sed_states.png")
