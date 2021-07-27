import click
import logging
from fitting import fit_state

logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


@click.group()
def cli():
    pass


@click.command("fit")
@click.option(
    "--state", type=click.Choice(["low", "2012", "2015a", "2015b"], case_sensitive=True)
)
@click.option("--k_e", default=1e-2, type=float, help="electorn normalisation")
@click.option("--gamma_min", default=1, type=float, help="maximum Lorentz factor")
@click.option("--gamma_max", default=5e4, type=float, help="maximum Lorentz factor")
@click.option("--t_var", default=0.5, type=float, help="variability time scale")
@click.option("--r", default=6e17, type=float, help="distance of blob from BH")
def fit(state, k_e, gamma_min, gamma_max, t_var, r):
    """perform the fit of PKS 1510-089 SED for a given state"""
    fit_state(state, k_e, gamma_min, gamma_max, t_var, r)


# add the commands
cli.add_command(fit)


# main execution
if __name__ == "__main__":
    cli()
