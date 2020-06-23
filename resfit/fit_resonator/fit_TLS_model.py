import attr
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import sympy as sym
from scipy.optimize import curve_fit
from matplotlib.patches import Circle
from lmfit import Minimizer
import inflect
import matplotlib.pylab as pylab
from scipy import optimize
from scipy import stats
import time
import sys
import os
pathToParent = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #set a variable that equals the relative path of parent directory
sys.path.append(pathToParent)#path to Fit_Cavity

from scipy.interpolate import interp1d

params = {'legend.fontsize': 10,
          'figure.figsize': (10, 8),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize':18,
         'ytick.labelsize':18,
         'lines.markersize' : 1,
         'lines.linewidth' : 2,
         'font.size': 15.0 }
pylab.rcParams.update(params)

np.set_printoptions(precision=4,suppress=True)
p = inflect.engine() # search ordinal

@attr.s

# done i think
class PowerSweepParams:
    """A container to hold outputs from fitting a resonator power sweep."""
    #power = attr.ib(type=np.ndarray)
    Qi = attr.ib(type=np.ndarray)
    QiCI = attr.ib(type=np.ndarray)
    Qc = attr.ib(type=np.ndarray)
    QcCI = attr.ib(type=np.ndarray)

    @classmethod
    def from_csv(cls, csv):
        """Load data from csv file."""
        try:
            data = np.loadtxt(csv, delimiter=',')
        except:
            print("User data file not found.")
            quit()
        #power = data.T[0]
        Qi = data.T[0]
        QiCI = data.T[1]
        Qc = data.T[2]
        QcCI = data.T[3]
        #return cls(power=power,Qi=Qi,QiCI=QiCI,Qc=Qc,QcCI=QcCI)
        return cls(Qi=Qi,QiCI=QiCI,Qc=Qc,QcCI=QcCI)

def estimate_photons(power: np.ndarray, lineattenuation: np.ndarray, Qi: np.ndarray, Qc, f0):
	
	hbar = 1.0545718E-34
	Ql = 1/(1/Qi + 1/Qc)
	Z0overZc = 1
	totalpower = [x + lineattenuation for x in power]

	return np.array((Z0overZc)*(2*(Ql^2)*(10^(totalpower/10)))/(hbar*(2*pi*f0)^2*Qc))

def PlotFit(photons,Qi,params,func,error,figurename,x_c,y_c,radius,output_path,conf_array,extract_factor = None,title = "Fit",manual_params = None):
    """Plots data and outputs fit parameters to a file

    Args:
        x: cropped frequency data
        y: cropped S21 data after normalization
        x_initial: original frequency data before normalization
        y_initial: original S21 data before normalization
        slope: slope of the normalization line for magnitude
        intercept: intercept of normalization line for magnitude
        slope2: slope of the normalization line for phase
        intercept2: intercept of normalization line for phase
        params: parameters generated from fit function
        Method: Method class instance
        func: function used to generate data for plotting (DCM/DCM REFLECTION/INV/PHI/CPZM)
        error: error from Monte Carlo Fit function
        figurename: name the plot should have
        x_c: center x coordinate for fit circle
        y_c: center y coordinate for fit circle
        radius: radius of fit circle
        output_path: file path where the plots will be saved
        conf_array: array with 95% confidence interval values
        extract_factor: contains starting and ending frequency values
        title: title the plot will have
        manual_params: user input manual initial guess parameters for fit

    Returns:
        plot output to file
    """
    #close plot if still open
    plt.close(figurename)
    #generate an even distribution of 5000 frequency points between the min and max of x for graphing purposes
    if extract_factor == None:
        x_fit = np.linspace(x.min(),x.max(),5000)
    elif isinstance(extract_factor, list) == True:
        x_fit = np.linspace(extract_factor[0],extract_factor[1],5000)
    #plug in the 5000 x points to respective fit function to create set of S21 data for graphing
    y_fit = func(x_fit,*params)

    fig = plt.figure(figurename,figsize=(15, 10))
    gs = GridSpec(6,6)
    #original magnitude
    ax1 = plt.subplot(gs[0:1,4:6])
    #original angle
    ax2 = plt.subplot(gs[2:3,4:6])
    #normalized magnitude
    ax3 = plt.subplot(gs[1:2,4:6])
    #normalized angle
    ax4 = plt.subplot(gs[3:4,4:6])
    #IQ plot
    ax = plt.subplot(gs[2:6,0:4])

    #add title
    if len(title) > 77:
        plot_title = title[0:39] + "\n" + title[40:76] + '...'
        plt.gcf().text(0.05, 0.9, plot_title, fontsize=30)
    if len(title) > 40:
        plot_title = title[0:39] + "\n" + title[40:]
        plt.gcf().text(0.05, 0.9, plot_title, fontsize=30)
    else:
        plt.gcf().text(0.05, 0.92, title, fontsize=30)

    #manual parameters
    textstr = ''
    if manual_params != None:
        if func == ff.cavity_inverse:
            textstr = r'Manually input parameters:' + '\n' + r'$Q_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[1])) + \
            '\n' + r'$Q_i$ = '+'%s' % float('{0:.5g}'.format(manual_params[0]))+\
            '\n' + r'$f_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[2]))+' GHz'\
            '\n' + r'$\phi$ = '+'%s' % float('{0:.5g}'.format(manual_params[3]))+' radians'
        elif func == ff.cavity_CPZM:
            textstr = r'Manually input parameters:' + '\n' + r'$Q_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[0]*manual_params[1]**-1)) + \
            '\n' + r'$Q_i$ = '+'%s' % float('{0:.5g}'.format(manual_params[0]))+\
            '\n' + r'$Q_a$ = '+'%s' % float('{0:.5g}'.format(manual_params[0]*manual_params[3]**-1))+\
            '\n' + r'$f_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[2]))+' GHz'
        else:
            Qc = manual_params[1]/np.exp(1j*manual_params[3])
            Qi = (manual_params[0]**-1-abs(np.real(Qc**-1)))**-1
            textstr = r'Manually input parameters:' + '\n' + 'Q = '+ '%s' % float('{0:.5g}'.format(manual_params[0])) + \
            '\n' + r'1/Re[1/$Q_c$] = ' +'%s' % float('{0:.5g}'.format(1/np.real(1/Qc))) + \
            '\n' + r'$Q_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[1])) + \
            '\n' + r'$Q_i$ = '+'%s' % float('{0:.5g}'.format(Qi))+\
            '\n' + r'$f_c$ = '+'%s' % float('{0:.5g}'.format(manual_params[2]))+' GHz'+\
            '\n' + r'$\phi$ = '+'%s' % float('{0:.5g}'.format(manual_params[3]))+' radians'
        plt.gcf().text(0.1, 0.7, textstr, fontsize=15)
    else:
        plt.gcf().text(0.05, 0.85, "No manually input parameters", fontsize=15)

    if isinstance(extract_factor, list) == True:
        x_fit_full = np.linspace(x.min(),x.max(),5000)
        y_fit_full = func(x_fit_full,*params)

        x_fit = np.linspace(extract_factor[0],extract_factor[1],5000)
        y_fit = func(x_fit,*params)

    if func == ff.cavity_inverse:
        ax1.set_ylabel('Mag[S21]')
        ax2.set_ylabel('Ang[S21]')
        ax3.set_ylabel('Mag[1/S21]')
        ax4.set_ylabel('Ang[1/S21]')
    else:
        ax1.set_ylabel('Mag[S21]')
        ax2.set_ylabel('Ang[S21]')
        ax3.set_ylabel('Mag[S21]')
        ax4.set_ylabel('Ang[S21]')

    ax1.plot(x_initial,np.log10(np.abs(y_initial))*20,'bo',label = 'raw data',color = 'black')
    ax1.plot(x,x*slope2+intercept2,'g-',label = 'normalize line',color = 'orange',linewidth = 1.5)
    ax1.set_xlim(left=x[0], right=x[-1])
    ax1.set_xlabel('frequency (GHz)')
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    ax2.plot(x_initial,np.angle(y_initial),'bo',label = 'raw data',color = 'black')
    ax2.plot(x,x*slope+intercept,'g-',label = 'normalize line',color = 'orange',linewidth = 1.5)
    ax2.set_xlim(left=x[0], right=x[-1])
    ax2.set_xlabel('frequency (GHz)')
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    ax3.plot(x,np.log10(np.abs(y))*20,'bo',label = 'normalized data')
    #ax3.plot(x_fit_full,np.log10(np.abs(y_fit_full))*20,'g--',label = 'fit past 3dB from resonance',color = 'lightgreen', alpha=0.7)
    ax3.plot(x_fit,np.log10(np.abs(y_fit))*20,'g-',label = 'fit function')
    ax3.set_xlim(left=x[0], right=x[-1])
    ax3.set_xlabel('frequency (GHz)')
    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)

    ax4.plot(x,np.angle(y),'bo',label = 'normalized data')
    #ax4.plot(x_fit_full,np.angle(y_fit_full),'g--',label = 'fit past 3dB from resonance',color = 'lightgreen', alpha=0.7)
    ax4.plot(x_fit,np.angle(y_fit),'g-',label = 'fit function')
    ax4.set_xlim(left=x[0], right=x[-1])
    ax4.set_xlabel('frequency (GHz)')
    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(12)


    line2 = ax.plot(np.real(y_fit),np.imag(y_fit),'g-',label = 'fit function',linewidth = 6)
    line1 = ax.plot(np.real(y),np.imag(y),'bo',label = 'normalized data',markersize = 2)
    #ax.plot(np.real(y_fit_full),np.imag(y_fit_full),'--',color = 'lightgreen',label = 'fit past 3dB from resonance',linewidth = 4.5, alpha=0.7)
    if x_c ==0 and y_c ==0 and radius == 0:
        pass
    else:
        plt.plot(x_c,y_c,'g-',markersize = 5,color=(0, 0.8, 0.8),label = 'initial guess circle')
        circle = Circle((x_c, y_c), radius, facecolor='none',\
                    edgecolor=(0, 0.8, 0.8), linewidth=3, alpha=0.5)
        ax.add_patch(circle)
    #plot resonance
    if func == ff.cavity_inverse:
        resonance = (1 + params[0]/params[1]*np.exp(1j*params[3])/(1 + 1j*2*params[0]*(params[2]-params[2])/params[2]))
    elif func == ff.cavity_DCM:
        resonance = 1-params[0]/params[1]*np.exp(1j*params[3])
    elif func == ff.cavity_DCM_REFLECTION:
        resonance = (1-2*params[0]/params[1]*np.exp(1j*params[3])/(1 + 1j*(params[2]-params[2])/params[2]*2*params[0]))
    elif func == ff.cavity_CPZM:
        resonance = 1/(1 + params[1] +1j*params[3])
    else:
        resonance = 1 + 1j*0
    ax.plot(np.real(resonance),np.imag(resonance),'*',color = 'red',label = 'resonance',markersize = 7)
    ax3.plot(params[2],np.log10(np.abs(resonance))*20,'*',color = 'red',label = 'resonance',markersize = 7)
    ax4.plot(params[2],np.angle(resonance),'*',color = 'red',label = 'resonance',markersize = 7)


    plt.axis('square')
    plt.ylabel('Im[S21]')
    plt.xlabel("Re[S21]")
    if func == ff.cavity_inverse:
         plt.ylabel('Im[$S_{21}^{-1}$]')
         plt.xlabel("Re[$S_{21}^{-1}$]")
    leg = plt.legend()
    ax1.legend(fontsize=8)

	# get the individual lines inside legend and set line width
    for line in leg.get_lines():
        line.set_linewidth(10)

    try:
        if params != []:
            if func == ff.cavity_inverse:
                if params[0] < 0:
                    print("Qi is less than zero. Please make sure data is of correct format: decibals (log10*20 version), and radians. Otherwise, it is quite likely that the resonator being fit is not a Notch type resonator. For reflection type geometry, please use DCM REFLECTION.")
                if conf_array[0] > 10**-10 and conf_array[0] != float('inf'):
                    Qi = params[0]-params[0]%(10**int(np.log10(conf_array[0])-1))
                else:
                    Qi = params[0]
                if conf_array[1] > 10**-10 and conf_array[1] != float('inf'):
                    Qc = params[1]-params[1]%(10**int(np.log10(conf_array[1])-1))
                else:
                    Qc = params[1]
                if conf_array[2] > 10**-10 and conf_array[2] != float('inf'):
                    phi = params[3]-params[3]%(10**int(np.log10(conf_array[2])-1))
                else:
                    phi = params[3]
                if conf_array[3] > 10**-10 and conf_array[3] != float('inf'):
                    f_c = params[2]-params[2]%(10**int(np.log10(conf_array[3])-1))
                else:
                    f_c = params[2]
                textstr = r'$Q_i$ = '+'%s' % float('{0:.10g}'.format(Qi))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[0]))+\
                '\n' + r'$Q_c^*$ = '+'%s' % float('{0:.10g}'.format(Qc))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                '\n' + r'$\phi$ = '+'%s' % float('{0:.10g}'.format(phi))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[2]))+' radians'+\
                '\n' + r'$f_c$ = '+'%s' % float('{0:.10g}'.format(f_c))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[3]))+' GHz'
                plt.gcf().text(0.7, 0.11, textstr, fontsize=18)
            elif func == ff.cavity_CPZM:
                if params[0] < 0:
                    print("Qi is less than zero. Please make sure data is of correct format: decibals (log10*20 version), and radians. Otherwise, it is quite likely that the resonator being fit is not a Notch type resonator. For reflection type geometry, please use DCM REFLECTION.")
                if conf_array[0] > 10**-10 and conf_array[0] != float('inf'):
                    Qi = params[0]-params[0]%(10**int(np.log10(conf_array[0])-1))
                else:
                    Qi = params[0]
                if conf_array[1] > 10**-10 and conf_array[1] != float('inf'):
                    Qc = (params[0]*params[1]**-1)-(params[0]*params[1]**-1)%(10**int(np.log10(conf_array[1])-1))
                else:
                    Qc = (params[0]*params[1]**-1)
                if conf_array[2] > 10**-10 and conf_array[2] != float('inf'):
                    Qa = (params[0]*params[3]**-1)-(params[0]*params[3]**-1)%(10**int(np.log10(conf_array[2])-1))
                else:
                    Qa = (params[0]*params[3]**-1)
                if conf_array[3] > 10**-10 and conf_array[3] != float('inf'):
                    f_c = params[2]-params[2]%(10**int(np.log10(conf_array[3])-1))
                else:
                    f_c = params[2]
                textstr = r'$Q_i$ = '+'%s' % float('{0:.10g}'.format(Qi))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[0]))+\
                '\n' + r'$Q_c$ = '+'%s' % float('{0:.10g}'.format(Qc))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                '\n' + r'$Q_a$ = '+'%s' % float('{0:.10g}'.format(Qa))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[2]))+\
                '\n' + r'$f_c$ = '+'%s' % float('{0:.10g}'.format(f_c))+ r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[3]))+' GHz'
                plt.gcf().text(0.7, 0.11, textstr, fontsize=18)
            else:
                Qc = params[1]/np.exp(1j*params[3])
                if Method.method == 'PHI':
                    Qi = (params[0]**-1-np.abs(Qc**-1))**-1
                else:
                    Qi = (params[0]**-1-np.real(Qc**-1))**-1

                if Qi < 0 and Method.method != 'DCM REFLECTION':
                    print("Qi is less than zero. Please make sure data is of correct format: decibals (log10*20 version), and radians. Otherwise, it is quite likely that the resonator being fit is not a Notch type resonator. Other types of resonators will not work with this code.")
                if 1/np.real(1/Qc) < 0:
                    print("Warning: 1/Real[1/Qc] is less than 0. Calculating Qi anyway")
                    if conf_array[0] > 10**-10 and conf_array[0] != float('inf'):
                        Q = params[0]-params[0]%(10**int(np.log10(conf_array[0])-1))
                    else:
                        Q = params[0]
                    if conf_array[1] > 10**-10 and conf_array[1] != float('inf'):
                        Qi = Qi-Qi%(10**int(np.log10(conf_array[1])-1))
                    if conf_array[2] > 10**-10 and conf_array[2] != float('inf'):
                        Qc = params[1]-params[1]%(10**int(np.log10(conf_array[2])-1))
                    else:
                        Qc = params[1]
                    if conf_array[3] > 10**-10 and conf_array[3] != float('inf'):
                        Qc_Re = (1/np.real(1/(params[1]/np.exp(1j*params[3]))))-(1/np.real(1/(params[1]/np.exp(1j*params[3]))))%(10**int(np.log10(conf_array[3])-1))
                    else:
                        Qc_Re = (1/np.real(1/(params[1]/np.exp(1j*params[3]))))
                    if conf_array[4] > 10**-10 and conf_array[4] != float('inf'):
                        phi = params[3]-params[3]%(10**int(np.log10(conf_array[4])-1))
                    else:
                        phi = params[3]
                    if conf_array[5] > 10**-10 and conf_array[5] != float('inf'):
                        f_c = params[2]-params[2]%(10**int(np.log10(conf_array[5])-1))
                    else:
                        f_c = params[2]
                    textstr = 'Q = '+ '%s' % float('{0:.10g}'.format(Q)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[0])) + \
                    '\n' + r'$Q_i$ = '+ '%s' % float('{0:.10g}'.format(Qi)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                    '\n' + r'$Q_c$ = '+ '%s' % float('{0:.10g}'.format(Qc)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[2])) + \
                    '\n' + r'$\phi$ = '+'%s' % float('{0:.10g}'.format(phi)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[4]))+' radians'+\
                    '\n' + r'$f_c$ = '+'%s' % float('{0:.10g}'.format(f_c)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[5]))+' GHz'
                    plt.gcf().text(0.7, 0.09, textstr, fontsize=18)
                    Qc_str = r'1/Re[1/$Q_c$] = ' +'%s' % float('{0:.10g}'.format(Qc_Re)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[3]))
                    plt.gcf().text(0.7, 0.245, Qc_str, fontsize=18, color = 'red')

                else:
                    if conf_array[0] > 10**-10 and conf_array[0] != float('inf'):
                        Q = params[0]-params[0]%(10**int(np.log10(conf_array[0])-1))
                    else:
                        Q = params[0]
                    if conf_array[1] > 10**-10 and conf_array[1] != float('inf'):
                        Qi = Qi-Qi%(10**int(np.log10(conf_array[1])-1))
                    if conf_array[2] > 10**-10 and conf_array[2] != float('inf'):
                        Qc = params[1]-params[1]%(10**int(np.log10(conf_array[2])-1))
                    else:
                        Qc = params[1]
                    if conf_array[3] > 10**-10 and conf_array[3] != float('inf'):
                        Qc_Re = (1/np.real(1/(params[1]/np.exp(1j*params[3]))))-(1/np.real(1/(params[1]/np.exp(1j*params[3]))))%(10**int(np.log10(conf_array[3])-1))
                    else:
                        Qc_Re = (1/np.real(1/(params[1]/np.exp(1j*params[3]))))
                    if conf_array[4] > 10**-10 and conf_array[4] != float('inf'):
                        phi = params[3]-params[3]%(10**int(np.log10(conf_array[4])-1))
                    else:
                        phi = params[3]
                    if conf_array[5] > 10**-10 and conf_array[5] != float('inf'):
                        f_c = params[2]-params[2]%(10**int(np.log10(conf_array[5])-1))
                    else:
                        f_c = params[2]
                    textstr = 'Q = '+ '%s' % float('{0:.10g}'.format(Q)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[0])) + \
                    '\n' + r'$Q_i$ = '+ '%s' % float('{0:.10g}'.format(Qi)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                    '\n' + r'$Q_c$ = '+ '%s' % float('{0:.10g}'.format(Qc)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[2])) + \
                    '\n' + r'1/Re[1/$Q_c$] = ' +'%s' % float('{0:.10g}'.format(Qc_Re)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[3])) + \
                    '\n' + r'$\phi$ = '+'%s' % float('{0:.10g}'.format(phi)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[4]))+' radians'+\
                    '\n' + r'$f_c$ = '+'%s' % float('{0:.10g}'.format(f_c)) + r"$\pm$" + '%s' % float('{0:.1g}'.format(conf_array[5]))+' GHz'
                    plt.gcf().text(0.7, 0.09, textstr, fontsize=18)

            #write to output csv file
            title_without_period = ''
            for i in title: #remove period from title
                if i != '.':
                    title_without_period = title_without_period + i
            file = open(output_path + "fit_params.csv","w")
            if func == ff.cavity_inverse:
                textstr = r'Qi = '+'%s' % float('{0:.10g}'.format(Qi))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[0]))+\
                '\n' + r'Qc* = '+'%s' % float('{0:.10g}'.format(Qc))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                '\n' + r'phi = '+'%s' % float('{0:.10g}'.format(phi))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[2]))+' radians'+\
                '\n' + r'fc = '+'%s' % float('{0:.10g}'.format(f_c))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[3]))+' GHz'
            elif func == ff.cavity_CPZM:
                textstr = r'Q_i = '+'{0:01f}'.format(params[0]) + \
                '\n' + r'Q_c = '+'{0:01f}'.format(params[0]*params[1]**-1)+\
                '\n' + r'Q_a = '+'{0:01f}'.format(params[0]*params[3]**-1)+\
                '\n' + r'f_c = '+'{0:01f}'.format(params[2])+' GHz'

                textstr = r'Qi = '+'%s' % float('{0:.10g}'.format(Qi))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[0]))+\
                '\n' + r'Qc = '+'%s' % float('{0:.10g}'.format(Qc))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                '\n' + r'Qa = '+'%s' % float('{0:.10g}'.format(Qa))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[2]))+\
                '\n' + r'fc = '+'%s' % float('{0:.10g}'.format(f_c))+ r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[3]))+' GHz'
            else:
                textstr = 'Q = '+ '%s' % float('{0:.10g}'.format(Q)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[0])) + \
                '\n' + r'Qi = '+ '%s' % float('{0:.10g}'.format(Qi)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[1]))+\
                '\n' + r'Qc = '+ '%s' % float('{0:.10g}'.format(Qc)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[2])) + \
                '\n' + r'1/Re[1/Qc] = ' +'%s' % float('{0:.10g}'.format(Qc_Re)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[3])) + \
                '\n' + r'phi = '+'%s' % float('{0:.10g}'.format(phi)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[4]))+' radians'+\
                '\n' + r'fc = '+'%s' % float('{0:.10g}'.format(f_c)) + r"+/-" + '%s' % float('{0:.1g}'.format(conf_array[5]))+' GHz'
        file.write(textstr)
    except:
        print(">Error when trying to write parameters on plot")
        quit()

    plt.tight_layout()
    return fig

def TLS_model(params: np.ndarray,photons: np.ndarray,tanh_hf0kbT):

	return np.array(((params(2)*tanh_hf0kbT)/(1+photons/params(3))^params(4) + 1/params(1)))

def min_fit(initguess,photons,Qi):
    """Minimizes parameter values

    Args:
        initguess: guess for correct values of the fit parameters
		photons:
		Qi:

    Returns:
        minimized parameter values, 95% confidence intervals for those parameter values
    """

    minner = Minimizer(TLS_model, initguess, fcn_args=(photons, Qi))
    result = minner.minimize(method = 'least_squares')

    fit_params = result.params
    parameter = fit_params.valuesdict()
    #extracts the actual value for each parameter and puts it in the fit_params list
    fit_params = [value for _,value in parameter.items()]

    ci = lmfit.conf_interval(minner, result, p_names=['QHP','FtandTLS','Nc','beta'], sigmas=[2])
    #confidence interval for QHP
    QHP_CI = max(np.abs(ci['QHP'][1][1]-ci['QHP'][0][1]),np.abs(ci['QHP'][1][1]-ci['QHP'][2][1]))
    #confidence interval for FtandTLS
    FtandTLS_CI = max(np.abs(ci['FtandTLS'][1][1]-ci['FtandTLS'][0][1]),np.abs(ci['FtandTLS'][1][1]-ci['FtandTLS'][2][1]))
    #confidence interval for Nc
    Nc_CI = max(np.abs(ci['Nc'][1][1]-ci['Nc'][0][1]),np.abs(ci['Nc'][1][1]-ci['Nc'][2][1]))
    #confidence interval for beta
    beta_CI = max(np.abs(ci['beta'][1][1]-ci['beta'][0][1]),np.abs(ci['beta'][1][1]-ci['beta'][2][1]))
    #Array of confidence intervals
    CI_array = [QHP_CI,FtandTLS_CI,Nc_CI,beta_CI]

    return fit_params, CI_array
    #except:
    #    print(">Failed to minimize function for least squares fit")
    #    quit()

def fit_powersweep(filename: str,dir: str, initguess: np.ndarray):
    """Function to fit a resonator power sweep to the TLS model

    Args:
        filename: name of the file to be fit
        dir: directory where data to be fit is contained
    Returns:

    """

    #read in data from file
    # power (dBm); Qi; Qi ci; Qc; Qc ci
    if dir != None:
        filepath = dir+'/'+filename
        data = PowerSweepParams.from_csv(filepath)
    else:
        print("No data was input. Please input a directory to read a file in.")
        quit()

    #separate data by column
    try:
        power = data.power
        Qi = data.Qi
        QiCI = data.QiCI
        Qc = data.Qc
        QcCI = data.QxCI
    except:
        print("Data unable to be read from PowerSweepParams class.")
        quit()

    #make a folder to put all output in
    result = time.localtime(time.time())
    output = str(result.tm_year)
    output = 'TLSfit_powersweep_' + output
    if len(str(result.tm_mon)) < 2:
        output = output + '0' + str(result.tm_mon)
    else:
        output = output + str(result.tm_mon)
    if len(str(result.tm_mday)):
        output = output + '0' + str(result.tm_mday) + '_' + str(result.tm_hour) + '_' + str(result.tm_min) + '_' + str(result.tm_sec)
    else:
        output = output + str(result.tm_mday) + '_' + str(result.tm_hour) + '_' + str(result.tm_min) + '_' + str(result.tm_sec)
    if dir != None:
        output_path = dir + '/' + output + '/'
    else:
        output_path = output + '/'
    os.mkdir(output_path)

    #Step Two. Fit Both Re and Im data
        # create a set of Parameters

    #define parameters from initial guess
    try:
        #initialize parameter class, min is lower bound, max is upper bound, vary = boolean to determine if parameter varies during fit
        params = lmfit.Parameters()
        params.add('QHP', value=initguess[0],vary = vary[0],min = initguess[0]*0.8, max = initguess[0]*1.2)
        params.add('FtandTLS', value=initguess[1],vary = vary[1],min = initguess[1]*0.8, max = initguess[1]*1.2)
        params.add('Nc', value=initguess[2],vary = vary[2],min = initguess[2]*0.9, max = initguess[2]*1.1)
        params.add('beta', value=initguess[3],vary = vary[3],min = initguess[3]*0.9, max = initguess[3]*1.1)
    except:
        print(">Failed to define parameters, please make sure parameters are of correct format")
        quit()

    #Fit data to least squares fit for respective fit type
    fit_params,CI_array = min_fit(initguess,photons,Qi)

    #plot fit
    figurename = "TLS fit of power sweep"
    fig = PlotFit(power,Qi,output_params,ff.cavity_DCM,error,figurename,x_c,y_c,r,output_path,CI_array, manual_params = Method.manual_init)

    fig.savefig(output_path+'TLS_fit_powersweep.png')

    return output_params,CI_array,fig,error,init