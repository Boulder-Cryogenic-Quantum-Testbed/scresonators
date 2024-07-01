# -*- encoding: utf-8 -*-
"""
User code to fit S21 data for the Mines Al 6061 3D Cavity
Data collection information:
---------------------------
Collector: Dave Pappas
VNA: Keysight PNA
Date collected: 210430
Purpose: Collect power and power dependence of cavity resonance
         with a SiOx sample loaded
---------------------------
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #speadsheet commands
import datetime
import sys #update paths
import os #import os in order to find relative path
import glob

def stitch_broadband(prefix, freq_band, freq_step, dstr, powers,
                    Tmxc=13, fscale=1e9, sparam='S21', show_plots=True,
                    manual_fnames=None, manual_dirs=None):
    """
    Stitch together multiple high power sweeps and plot as a single file
    """
    # Read the frequency band and powers used
    f1, f2 = freq_band
    p1 = powers

    if manual_fnames is None:
        # Determine number of frequency segments
        Nf = int(round((f2 - f1) / freq_step))
        center_freqs = [freq_band[0] + (1 + 2*j) * freq_step / 2 
                        for j in range(Nf)]
        center_freqs_strs = [f'{cf:.3f}'.replace('.', 'p') for cf in center_freqs]
        print(f'center_freqs: {center_freqs:.3f}')
        print(f'center_freqs_strs: {center_freqs_strs:.3f}')

        # Generate directories and paths
        sdirs = [f'{prefix}_{cf}GHz*'
                for cf in center_freqs_strs]
        print(f'sdirs:\n{sdirs}')
        dirs = [glob.glob(f'{prefix}_{cf}GHz*')[0]
                for cf in center_freqs_strs]
        print(f'sdirs:\n{sdirs}')
        fnames = [f'{d}/{prefix}_{cff}GHz_{p1:.0f}dB_{Tmxc:.0f}mK.csv'
                    for cff, d in zip(center_freqs_strs, dirs)]

    else:
        fnames = [ data_dir + "\\" + fname for fname, data_dir in zip(manual_fnames, manual_dirs)]
        
    # Read the data and concatenate
    freqs  = np.array([])
    S21mag = np.array([])
    S21ph  = np.array([])
    for fn in fnames:
        data = np.genfromtxt(fn, delimiter=',').T
        freqs = np.hstack((freqs, data[0]))
        S21mag = np.hstack((S21mag, data[1]))
        S21ph = np.hstack((S21ph, data[2]))

    # Plot the results
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    fsize = 20
    ax.plot(freqs / fscale, S21mag)
    ax.set_xlabel('Frequency [GHz]', fontsize=fsize)
    ax.set_ylabel(r'$|S_{%s}|$' % sparam[1:], fontsize=fsize)
    
    plot_fname1 = f'{prefix}_{f1:.2f}GHz_{f2:.2f}GHz_magn'.replace('.','p')
    plt.title(plot_fname1)
    fig.savefig(plot_fname1+'.png', format='png')
    
    if show_plots is False:
        plt.close('all')
    
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    fsize = 20
    ax.plot(freqs / fscale, np.unwrap(S21ph))
    ax.set_xlabel('Frequency [GHz]', fontsize=fsize)
    ax.set_ylabel(r'$\left< S_{%s} \right.$' % sparam[1:], fontsize=fsize)
    
    plot_fname2 = f'{prefix}_{f1:.2f}GHz_{f2:.2f}GHz_phase'.replace('.','p')
    plt.title(plot_fname2)
    fig.savefig(plot_fname2+'.png', format='png')
    
    if show_plots is False:
        plt.close('all')


if __name__ == '__main__':
    
    # Concatenate broadband sweeps
    prefix = 'NWNb2O5_3_01'

    fcs = [5.8, 6.3]
    spans = [500]*len(fcs)
    freq_band = [fcs[0], fcs[-1]]
    freq_step = spans[0]*1e-3

    dstr = '240524' 
    powers = [-5]
    
    # manual_dirs=["NWNb2O5_3_01_5.400GHz_6.400GHz", "NWNb2O5_3_01_5.900GHz_6.900GHz"]
    # manual_fnames=["NWNb2O5_3_01_5p900GHz_-5dB_12mK_.csv","NWNb2O5_3_01_6p400GHz_-5dB_12mK_.csv"]
    
    manual_dirs = glob.glob("*GHz")
    manual_fnames = []
    for directory in manual_dirs:
        manual_fnames.append(glob.glob(directory + "\\*.csv")[0].replace(directory+"\\",""))
    
    stitch_broadband(prefix, freq_band, freq_step, dstr, powers,
                    Tmxc=24., fscale=1e9, sparam='S12', 
                    manual_fnames=manual_fnames, manual_dirs=manual_dirs)
    
    
    stitch_broadband(prefix, [fcs[0]-spans[0]/2*1e-3,fcs[0]+spans[0]/2*1e-3], freq_step, dstr, powers,
                    Tmxc=24., fscale=1e9, sparam='S12', 
                    manual_fnames=[manual_fnames[0]], manual_dirs=[manual_dirs[0]])
    
    
    stitch_broadband(prefix, [fcs[1]-spans[1]/2*1e-3,fcs[1]+spans[1]/2*1e-3], freq_step, dstr, powers,
                    Tmxc=24., fscale=1e9, sparam='S12', 
                    manual_fnames=[manual_fnames[1]], manual_dirs=[manual_dirs[1]])
