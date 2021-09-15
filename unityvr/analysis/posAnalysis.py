import numpy as np
import pandas
import scipy as sp
import scipy.signal
import matplotlib.pyplot as plt

from unityvr.viz import viz
from unityvr.analysis.utilityFunctions import carryAttrs

from os.path import sep, exists, join

##functions to process posDf dataframe

#obtain the position dataframe with derived quantities
def position(uvrDat, derive = True, rotate_by = None, filter_date = '2021-09-08', plot = False, plotsave=False, saveDir=None):
    
    ## input arguments
    
    # set derive = True if you want to compute derived quantities (ds, s, dTh (change in angle), radangle (angle in radians(-pi,pi))) 
    
    # rotate_by: angle (degrees) by which to rotate the trajectory to ensure the bright part of the panorama is at 180 degree heading.
    
    # filter_date: date of experiment after which right handed angle convention will not be forced when loading posDf; this is because converting from Unity's left handed angle convention to right handed convention was implemented after a certain date in the preproc.py file
    
    #correct_convention: set to True if you want to correct the angle convention for preprocessed data
    
    posDf = uvrDat.posDf

    #angle correction
    #this is required only for data that was preprocessed before the filter_date
    if (np.datetime64(uvrDat.metadata['date'])<=np.datetime64(filter_date)) & ('angle_convention' not in uvrDat.metadata):
        print('correcting for Unity angle convention.')
        posDf['angle'] = (-posDf['angle'])%360
        uvrDat.metadata['angle_convention'] = "right-handed"

    #rotate
    if rotate_by is not None:
        posDf['x'], posDf['y'] = rotation_deg(posDf['x'],posDf['y'],rotate_by)
        posDf['dx'], posDf['dy'] = rotation_deg(posDf['dx'],posDf['dy'],rotate_by)
        posDf['dxattempt'], posDf['dyattempt'] = rotation_deg(posDf['dxattempt'],posDf['dyattempt'],rotate_by)
        posDf['angle'] = (posDf['angle']+rotate_by)%360
        uvrDat.metadata['rotated_by'] = rotate_by

    #add dc2cm conversion factor
    posDf.dc2cm = 10

    if derive:
        posDf['ds'] = np.sqrt(posDf['dx']**2+posDf['y']**2)
        posDf['s'] = np.cumsum(posDf['ds'])
        posDf['dTh'] = (np.diff(posDf['angle'],prepend=posDf['angle'].iloc[0]) + 180)%360 - 180
        posDf['radangle'] = ((posDf['angle']+180)%360-180)*np.pi/180
        
    if plot:
        fig, ax = viz.plotTrajwithParameterandCondition(posDf, figsize=(10,5), parameter='angle')
        if plotsave:
            fig.savefig(getTrajFigName("walking_trajectory",saveDir,uvrDat.metadata))

    return posDf

#segment flight bouts
def flightSeg(posDf, thresh, freq=120, plot = False, plotsave=False, saveDir=None, uvrDat=None):

    df = posDf.copy()

    #get spectrogram
    _, t, F = sp.signal.spectrogram(df['ds'], freq)

    # 2nd row of the spectrogram seems to contain sufficient information to segment flight bouts
    flight = sp.interpolate.interp1d(t,F[1,:]>thresh, kind='nearest', bounds_error=False)
    df['flight'] = flight(df['time'])

    #carry attributes
    df = carryAttrs(df,posDf)
    
    if plot:
        fig0, ax0 = plt.subplots()
        ax0.plot(t,F[1,:],'k');
        ax0.plot(df['time'],df['flight']*F[1,:].max(),'r',alpha=0.2);
        ax0.set_xlabel("time"); plt.legend(["power in HF band","thresholded"])
        
        fig1, ax1 = viz.plotTrajwithParameterandCondition(df, figsize=(10,5), parameter='angle', 
                                                        condition = (df['flight']==0))
        if plotsave:
            fig0.savefig(getTrajFigName("FFT_flight_segmentation",saveDir,uvrDat.metadata))
            fig1.savefig(getTrajFigName("walking_trajectory_segmented",saveDir,uvrDat.metadata))

    return df

#clip the dataframe
def flightClip(posDf, minT = 0, maxT = 485, plot = False, plotsave=False, saveDir=None, uvrDat=None):

    df = posDf.copy()

    #clip the position values according to the minT and maxT
    df['clipped'] = ((posDf['time']<=minT) | (posDf['time']>=maxT))

    #carry attributes
    df = carryAttrs(df,posDf)
    
    if plot:
        fig, ax = viz.plotTrajwithParameterandCondition(df, figsize=(10,5), parameter='angle', 
                                                        condition = (df['clipped']==0))
        if plotsave:
            fig.savefig(getTrajFigName("walking_trajectory_clipped",saveDir,uvrDat.metadata))

    return df

#rotation matrix for an Nx2 vector with [x,y]
def rotation_mat_rad(theta):
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return R

#rotate the trajectory by angle theta
def rotation_deg(x,y,theta):
    theta = np.pi/180*theta
    r = np.matmul(np.array([x,y]).T,rotation_mat_rad(theta))
    return r[:,0], r[:,1]

#add the derived quantities and clipping information to the saved dataframe
def posDfUpdate(posDf, uvrDat, saveDir, saveName):

    #update uvrDat
    uvrDat.posDf = posDf
    savepath = uvrDat.saveData(saveDir, saveName)
    print("location:", saveDir)


    