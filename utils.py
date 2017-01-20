# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scp import SCPClient
import os
import paramiko

def connect():
    # Open credentials file:
    fcred = open('credentials.dat','r')

    # Connect with SSH server:
    username,password,server,path = fcred.readline().split()
    fcred.close()
    transport = paramiko.Transport((server,22))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    # Return sftp and transport objects:
    return sftp,transport,path

import pyfits
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter
def getflux(fname):
    # Get flux, take nans out:
    d,h = pyfits.getdata(fname,header=True)
    t = d['TIME']+h['BJDREFI']
    flux = d['EVE_FLUX']
    if len(flux.shape) != 1:
        flux = flux[:,0]
    idx = ~np.isnan(flux)
    t,flux = t[idx],flux[idx]
    # Median-gaussian filter the lightcurve:
    window = int(np.sqrt(len(flux)))
    if window % 2 == 0:
        window += 1
    filt = gaussian_filter(medfilt(flux,window),5)
    # Return times and normalized flux:
    return t,flux/filt

def get_phases(t,P,t0):
    phase = ((t - np.median(t0))/np.median(P)) % 1
    ii = np.where(phase>=0.5)[0]
    phase[ii] = phase[ii]-1.0
    return phase

def grid_search(t,f,P,step = 0.02,in_phase = 10):
    # Default step is 0.02 days (30 minutes), and it 
    # is assumed 10 central points are representative 
    # of the deepest in-transit points:
    possible_t0 = np.arange(t[0],t[0]+P,0.02)
    depths = np.zeros(len(possible_t0))
    for i in range(len(possible_t0)):
        phases = get_phases(t,P,possible_t0[i])
        idx_in_transit = np.argsort(np.abs(phases))[:in_phase]
        depths[i] = 1.-np.median(f[idx_in_transit])
    idx_max = np.where(np.max(depths)==depths)[0]
    if len(idx_max)>1:
        idx_max = idx_max[0]
    return possible_t0[idx_max],depths[idx_max]

def box_model(t,ting,tin,delta):
    """
    This box model assumes t is centered on zero (ie phased), and units 
    of t, ting and tin are the same.
    """
    dur = tin + 2.*ting
    model = np.ones(len(t))
    idx = np.where(np.abs(t)<(tin/2.))[0]
    model[idx] = 1-delta
    idx = np.where((t>tin/2.)&(t<dur/2.))[0]
    b = delta/ting
    a = 1. - delta - (b*tin/2.)
    model[idx] = a + b*t[idx]
    idx = np.where((t<-tin/2.)&(t>-dur/2.))[0]
    b = -delta/ting
    a = 1. - delta + (b*tin/2.)
    model[idx] = a + b*t[idx]
    return model

import emcee
log2pi = np.log(2.*np.pi)
def box_mcmc(t,f,P_init,t0_init,depth_init,dur_init,nwalkers=500,nsteps=300,nburnin=300,name=None):
    def lnlike(p,t,y):
        P,t0,depth,ting,tin,sigma_w = p
        phases = get_phases(t,P,t0)*P
        residuals = (y - box_model(phases,ting,tin,depth*1e-6))*1e6
        taus = 1./sigma_w**2
        log_like = -0.5*(len(t)*log2pi+np.sum(np.log(1./taus)+taus*(residuals**2)))
        return log_like

    def lnprior(p):
        P,t0,depth,ting,tin,sigma_w = p
        if (P_init-0.1 < P < P_init+0.1) and (t0_init-0.1 < t0 < t0_init+0.1) and \
           (1. < depth < 1e6) and (0. < ting < 0.5) and (0.< tin < 0.5) \
           and (1. < sigma_w < 1e4):
            return 0.0
        return -np.inf

    def lnprob(p,t,y):
        lp = lnprior(p)
        return lp + lnlike(p,t,y) if np.isfinite(lp) else -np.inf

    if not os.path.exists('results/'+name):
        ting_init = (dur_init/4.)
        tin_init = (dur_init/2.)
        initial_parameters = np.array([P_init,t0_init,depth_init*1e6,ting_init,tin_init,300.])
        ndim = len(initial_parameters)
        # Define initial positions of the walkers:
        p0 = []
        for i in range(nwalkers):
            p0.append([initial_parameters[0] + np.random.normal(0,1e-5),\
                   initial_parameters[1] + np.random.normal(0,1e-5),\
                   initial_parameters[2] + np.random.normal(0,1e-5),\
                   np.random.uniform(0,0.5),\
                   np.random.uniform(0,0.5),\
                   np.random.uniform(1.,1e4)])
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,args=(t,f))
        sampler.reset()
        print '\t Running MCMC...' 
        sampler.run_mcmc(p0, nburnin+nsteps)
        print '\t ...done!'
        samples = np.zeros([nsteps*nwalkers,ndim])
        for i in range(ndim):
            dummy_array = np.array([])
            for walker in range(nwalkers):
                dummy_array = np.append(dummy_array,sampler.chain[walker,nburnin:,i])
            samples[:,i] = np.copy(dummy_array)
        P = samples[nburnin:,0]
        t0 = samples[nburnin:,1]
        depth = samples[nburnin:,2]
        ting = samples[nburnin:,3]
        tin = samples[nburnin:,4]
        sigma_w = samples[nburnin:,5]
        dur = tin + 2*ting
        if name is not None:
            if not os.path.exists('results'):
                os.mkdir('results')
            os.mkdir('results/'+name)
            pyfits.PrimaryHDU(P).writeto('results/'+name+'/P.fits')
            pyfits.PrimaryHDU(t0).writeto('results/'+name+'/t0.fits')
            pyfits.PrimaryHDU(depth).writeto('results/'+name+'/depth.fits')
            pyfits.PrimaryHDU(ting).writeto('results/'+name+'/ting.fits')
            pyfits.PrimaryHDU(tin).writeto('results/'+name+'/tin.fits')
            pyfits.PrimaryHDU(sigma_w).writeto('results/'+name+'/sigma_w.fits')
            pyfits.PrimaryHDU(dur).writeto('results/'+name+'/dur.fits')
    else:
        P = pyfits.getdata('results/'+name+'/P.fits')
        t0 = pyfits.getdata('results/'+name+'/t0.fits')
        depth = pyfits.getdata('results/'+name+'/depth.fits')
        ting = pyfits.getdata('results/'+name+'/ting.fits')
        tin = pyfits.getdata('results/'+name+'/tin.fits')
        sigma_w = pyfits.getdata('results/'+name+'/sigma_w.fits')
        dur = pyfits.getdata('results/'+name+'/dur.fits')
    return np.median(P),np.sqrt(np.var(P)), np.median(t0),np.sqrt(np.var(t0)),\
           np.median(depth),np.sqrt(np.var(depth)), np.median(ting),np.sqrt(np.var(ting)),\
           np.median(tin),np.sqrt(np.var(tin)), np.median(sigma_w),np.sqrt(np.var(sigma_w)),\
           np.median(dur),np.sqrt(np.var(dur))
