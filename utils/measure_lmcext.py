import warnings
import argparse
import numpy as np
import astropy.units as u
from astropy.table import QTable

from astropy.modeling.fitting import LevMarLSQFitter
from dust_extinction.shapes import FM90_B3

import matplotlib.pyplot as plt

from measure_extinction.stardata import StarData
from measure_extinction.extdata import ExtData

from models_mcmc_extension import EmceeFitter


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "redstar",
        help="Name of the reddened star",
    )
    parser.add_argument(
        "compstar",
        help="Name of the comparison star",
    )
    parser.add_argument(
        "--nsteps",
        type=int,
        default=1000,
        help="Number of steps for MCMC fitting",
    )
    parser.add_argument(
        "--burnfrac",
        type=float,
        default=0.5,
        help="Fraction of steps to discard as burn-in",
    )   
    args = parser.parse_args()

    # pick the fitter
    fit = LevMarLSQFitter()

    red_dat_filename = f"{args.redstar}.dat"
    comp_dat_filename = f"{args.compstar}.dat"
    dpath = "/home/kgordon/Python/extstar_data/LMC/" 

    redstar = StarData(red_dat_filename, path=dpath)
    compstar = StarData(comp_dat_filename, path=dpath)

    extdata = ExtData()
    extdata.calc_elx(redstar, compstar)

    extdata.calc_AV()
    extdata.calc_RV()

    ofile = f"LMCext/{args.redstar}_{args.compstar}_elv.fits"
    extdata.save(ofile)

    extdata.trans_elv_alav()
    print(extdata.columns)

    fontsize = 20
    font = {"size": fontsize}
    plt.rc("font", **font)
    plt.rc("lines", linewidth=2)
    plt.rc("axes", linewidth=2)
    plt.rc("xtick.major", width=2)
    plt.rc("ytick.major", width=2)

    fig, ax = plt.subplots(figsize=(15, 12), layout="constrained")

    extdata.plot(ax)
    ax.set_xscale("log")

    # fit extinction curve
    x = 1.0 / extdata.waves["IUE"]
    npts = extdata.npts["IUE"]
    npts[np.logical_and(3.55 / u.micron <= x, x < 3.58 / u.micron)] = 0
    npts[np.logical_and(3.83 / u.micron <= x, x < 3.90 / u.micron)] = 0
    npts[np.logical_and(4.17 / u.micron <= x, x < 4.22 / u.micron)] = 0
    npts[np.logical_and(4.25 / u.micron <= x, x < 4.28 / u.micron)] = 0
    npts[np.logical_and(6.4 / u.micron <= x, x < 6.6 / u.micron)] = 0
    npts[np.logical_and(7.15 / u.micron <= x, x < 7.25 / u.micron)] = 0
    npts[np.logical_and(7.45 / u.micron <= x, x < 7.55 / u.micron)] = 0
    npts[np.logical_and(7.62 / u.micron <= x, x < 7.74 / u.micron)] = 0
    npts[np.logical_and(7.90 / u.micron <= x, x < 7.95 / u.micron)] = 0
    npts[8.70 / u.micron <= x] = 0

    gvals = npts > 0
    x = 1.0 / extdata.waves["IUE"].value[gvals]
    y = extdata.exts["IUE"][gvals]
    weights = 1.0 / extdata.uncs["IUE"][gvals]

    # initialize the model
    fm90_init = FM90_B3()

    fm90_init.xo.bounds = (4.4, 4.9)
    fm90_init.xo = 4.59
    # fm90_init.xo.fixed = True
    fm90_init.gamma.bounds = (0.6, 1.7)
    fm90_init.gamma = 0.95
    # fm90_init.gamma.fixed = True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fm90_fit = fit(fm90_init, x, y, weights=1.0/extdata.uncs["IUE"][gvals])

        ax.plot(1/x, fm90_fit(x), color="black", label="FM90 fit")

    # Set up the backend to save the samples for the emcee runs
    emcee_samples_file = f"LMCext/{args.redstar}_{args.compstar}_alav_emcee_samples.h5"

    nsteps = args.nsteps
    fit3 = EmceeFitter(
        nsteps=nsteps, burnfrac=args.burnfrac, save_samples=emcee_samples_file
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        fm90_fit3 = fit3(fm90_fit, x, y, weights=weights)

    print("autocorr tau = ", fit3.fit_info["sampler"].get_autocorr_time(quiet=True))

    # setup the parameters for saving
    fm90_best_params = (fm90_fit.param_names, fm90_fit.parameters)
    fm90_per_param_vals = zip(
        fm90_fit3.parameters, fm90_fit3.uncs_plus, fm90_fit3.uncs_minus
    )
    fm90_per_params = (fm90_fit3.param_names, list(fm90_per_param_vals))

    fit_params = {}

    otab = QTable()
    otab["name"] = fm90_fit.param_names
    otab["value"] = fm90_fit.parameters
    fit_params["MIN"] = otab

    otab2 = QTable()
    otab2["name"] = fm90_fit3.param_names
    otab2["value"] = fm90_fit3.parameters
    otab2["unc"] = 0.5 * (fm90_fit3.uncs_plus + fm90_fit3.uncs_minus)
    fit_params["MCMC"] = otab2

    print(fit_params)
 
    ofile = f"LMCext/{args.redstar}_{args.compstar}_alav.fits"
    extdata.save(
        ofile,
        fit_params=fit_params,
    )

    # make the standard mcmc plots
    fit3.plot_emcee_results(fm90_fit3, filebase=ofile.replace(".fits", ""))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        ax.plot(1/x, fm90_fit3(x), label="emcee")

    plt.show()