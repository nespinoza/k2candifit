# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import utils
import os

first_connection = True
epicids,periods = np.loadtxt('candidates.dat',unpack=True)
prefix = 'k2_'
sufix = '-c102_llc.fits.gz'

# Iterate through each candidate:
for i in range(len(epicids)):
    epicid = epicids[i]
    # Get the filename:
    filename = prefix+str(int(epicid))+sufix
    # If filename not on datadir, download it from server:
    if not os.path.exists('datadir/'+filename):
        # If not connected to the sftp, connect:
        if first_connection:
            sftp,transport,path = utils.connect()
            first_connection = False
        # Get file from server:
        sftp.get(path+'/'+filename, 'datadir/'+filename)
    # Get the data:
    t,f = utils.getflux('datadir/'+filename)
    # Small grid search to get initial t0 and depth:
    t0_init,depth_init = utils.grid_search(t,f,periods[i])
    # Check:
    #phases = utils.get_phases(t,periods[i],t0_init)
    #plt.plot(phases,f,'.')
    #plt.plot(phases,np.ones(len(phases))-depth_init,'-')
    #plt.show()
    # Now fit the box:
    print '\t ------------------------------'
    print '\t Working on '+str(int(epicid))
    print '\t ------------------------------'
    print '\t Initial parameters:'
    print '\t P (days)      :',periods[i]
    print '\t t0 (days)     :',t0_init
    print '\t depth (ppm)   :',depth_init*1e6
    print '\t dur (days)    :',0.1
    print '\t ------------------------------'
    P,Perr,t0,t0err,depth,deptherr,ting,tingerr,tin,tinerr,sigma_w,sigma_werr,\
    dur,durerr = utils.box_mcmc(t,f,periods[i],t0_init,depth_init,0.1,name=str(int(epicid)))
    print '\t Final Parameters:'
    print '\t P (days)      :',P,'+-',Perr
    print '\t t0 (days)     :',t0,'+-',t0err
    print '\t depth (ppm)   :',depth,'+-',deptherr
    print '\t ting (days)   :',ting,'+-',tingerr
    print '\t tin (days)    :',tin,'+-',tinerr
    print '\t dur (days)    :',dur,'+-',durerr
    print '\t sigma_w (ppm) :',sigma_w,'+-',sigma_werr
    '\t ------------------------------'
    # Check fit:
    phases = utils.get_phases(t,P,t0)*P
    idx_order = np.argsort(phases)
    plt.plot(phases,f,'.')
    phase_model = np.linspace(-0.5,0.5,1e4)
    phase_model[0] = np.min(phases)
    phase_model[-1] = np.max(phases)
    model = utils.box_model(phase_model,ting,tin,depth*1e-6)
    plt.plot(phase_model,model,'-',linewidth=3,alpha=0.7)
    plt.xlim([-dur*2.,dur*2.])
    plt.ylim([1.-(depth*1e-6)*1.5,1.+sigma_w*5*1e-6])
    if not os.path.exists('results/plots'):
        os.mkdir('results/plots')
    plt.xlabel('Time (days since mid-transit)')
    plt.ylabel('Relative flux')
    plt.savefig('results/plots/'+str(int(epicid))+'.png')
    plt.clf()
    print '\n'

if not first_connection:
    sftp.close()
    transport.close()
