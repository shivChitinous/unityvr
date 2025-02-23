import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from unityvr.analysis.utils import getTrajFigName
from unityvr.viz import viz

from scipy.special import i0, iv
from scipy.optimize import curve_fit
import scipy.stats as sts


## Functions to fit data to heading direction distributions using von Mieses as model

#von mises probability density function
def vonmises_pdf(x, mu, kappa):
    #x, mu are in radians
    V = np.exp((kappa)*np.cos((x-mu)))/(2*np.pi*i0(kappa))
    return V

def sum_of_vonmises_pdf(x, mu1, mu2, kappa):
    #mixture of vonmises function with 4 parameters
    V = 0.5*(vonmises_pdf(x, mu1, kappa)+vonmises_pdf(x, mu2, kappa))
    return V

#function to fit data to the von mises pdf
def fit_vonmises(degAngles, binwidth = 20, plot = False, plotsave=False, saveDir=None, uvrDat=None,MFev=2000):
    # in degrees

    #width to radians
    binwidth = binwidth*np.pi/180

    #number of bins
    numbins = int(2*np.pi/binwidth)

    #convert to radians
    angles = degAngles*np.pi/180

    #get probability density and theta vector
    theta = np.linspace(0,2*np.pi,num=numbins+1)[:-1] + binwidth/2
    p = np.histogram(angles,bins=numbins,density=True)[0]

    headingPVAmag = np.abs(np.nanmean(np.exp(1j*theta)))

    plt.figure(figsize = (9,2))
    plt.step(theta*180/np.pi, p)

    #fit p as a function of theta
    params, _ = curve_fit(vonmises_pdf, theta, p, bounds=([0,0],[2*np.pi,np.inf]),maxfev=MFev)
    fit_func = vonmises_pdf(theta, params[0], params[1])

    #compute kolmogorov-smirnoff stat
    [ks, p_value] = sts.ks_2samp(p, fit_func)

    #compute squared difference from fit
    sqd = np.sum(np.square(p-fit_func))

    #decide whether distribution is unimodal
    notFit = ~np.any([p_value>0.1, headingPVAmag>0.5])

    #create a neitherFit variable
    neitherFit = False

    #if not unimodal:
    #fit p as a function of theta to a sum of vonmises
    if notFit:
        params, _ = curve_fit(sum_of_vonmises_pdf, theta, p, bounds=([0,0,0],[2*np.pi,2*np.pi,np.inf]),maxfev=MFev)
        fit_func = sum_of_vonmises_pdf(theta, params[0], params[1], params[2])

        #compute kolmogorov-smirnoff stat
        [ks, p_value] = sts.ks_2samp(p, fit_func)

        #compute squared difference from fit
        sqd = np.sum(np.square(p-fit_func))

        #decide whether distribution is bimodal:
        neitherFit =  ~(p_value>0.1)

    if neitherFit:
        print("Neither unimodal nor bimodal fit.")
        mu1 = float("NaN")
        mu2 =  float("NaN")
        kappa = float("NaN")
        p_value =  float("NaN")
        sqd = float("NaN")

    else:
        if notFit:
            mu1 = params[0]
            mu2 = params[1]
            kappa = params[2]

        else:
            mu1 = params[0]
            mu2 = None
            kappa = params[1]

        if plot:
            if notFit: V = sum_of_vonmises_pdf(np.linspace(0,2*np.pi,num=50), mu1, mu2, kappa)
            else: V = vonmises_pdf(np.linspace(0,2*np.pi,num=50), mu1, kappa)

            #plot 1
            plt.plot(np.linspace(0,360,num=50), V, 'k-')
            plt.xlabel(r"$\theta$")

            if plotsave:
                plt.savefig(getTrajFigName("fit_vonmises",saveDir,uvrDat.metadata))

            #plot 2
            fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
            ax.plot(mu1, kappa, 'ro', alpha=0.5)
            if notFit:
                ax.plot(mu2, kappa, 'bo', alpha=0.5)
            ax.set_yticks([0.5,1])
            ax.set_theta_zero_location("E")
            ax.set_xticks(np.pi/180 * np.arange(-180,  180,  45))
            ax.set_thetalim(-np.pi, np.pi);

            if plotsave:
                fig.savefig(getTrajFigName("mu_kappa",saveDir,uvrDat.metadata))

            #returns mu in degree

    return mu1*180/np.pi, mu2*180/np.pi if mu2 is not None else mu2, kappa, ks, p_value, sqd
