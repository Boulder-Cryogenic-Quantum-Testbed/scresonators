import sys
import matplotlib.pyplot as plt
import skrf as rf
import numpy as np

sys.path.append("..")
sys.path.append("../examples")
from src.utilities import unpack_s2p_df

#It might make sense to package all this into a subclass of matplotlib.Figure
#TODO: need to improve label placement when specified by user

def plot_s2p_df(s2p_df, plot_complex = True, zero_lines=False, track_min=False, plot_dict={}):
    
    ########### default settings ###########
    marker = plot_dict["marker"] if "marker" in plot_dict else "."    
    linestyle = plot_dict["linestyle"] if "linestyle" in plot_dict else None    
    plot_title = plot_dict["plot_title"] if "plot_title" in plot_dict else "VNA Data"    
    magn_color = plot_dict["magn_color"] if "magn_color" in plot_dict else "r"   
    phase_color = plot_dict["phase_color"] if "phase_color" in plot_dict else "b"  
    magn_label = plot_dict["magn_label"] if "magn_label" in plot_dict else None   
    phase_label = plot_dict["phase_label"] if "phase_label" in plot_dict else None  
    alpha = plot_dict["alpha"] if "alpha" in plot_dict else 1.0
    
    
    s2p_dict = unpack_s2p_df(s2p_df)
    
    freqs = s2p_dict["Frequency"]
    magn_lin,  magn_dB   =  s2p_dict["magn_lin"],   s2p_dict["magn_dB"]
    phase_rad, phase_deg =  s2p_dict["phase_rad"],  s2p_dict["phase_deg"]
    real,      imag      =  s2p_dict["real"],       s2p_dict["imag"]
    
    
    ########### begin plotting ###########
    mosaic = "AACC\nBBCC"
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(20,12))
    ax1, ax2, ax3 = axes["A"], axes["B"], axes["C"]
    axes = [ax1, ax2, ax3]
    
    ax1.grid()
    ax2.grid()
    
    if track_min is True:
        freq_argmin = magn_lin.argmin()
        freq_min = freqs[freq_argmin]
        new_freqs = (freqs - freq_min)/1e3
        freq_min_label = f"$f_{"{min}"}$ = {freq_min/1e6:1.3f} MHz"
        ax1.axvline(0, linestyle='--', color='k', linewidth=1)
        ax2.axvline(0, linestyle='--', color='k', linewidth=1)
        if ax3 is not None:
            ax3.plot(real[freq_argmin], imag[freq_argmin], 'y*', markersize=8, label=freq_min_label)
        fig.legend(loc="upper left")
        ax1.set_xlabel("Frequency $\\Delta f$ [MHz]")
        ax2.set_xlabel("Frequency $\\Delta f$ [MHz]")
        ax1.set_title(f"Freq vs Magn [$f_{"{min}"}$ = {freq_min/1e9:1.6f} GHz]")
        ax2.set_title(f"Freq vs Phase [$f_{"{min}"}$ = {freq_min/1e9:1.6f} GHz]")
    else:
        new_freqs = freqs
        ax1.set_xlabel("Frequency [GHz]")
        ax2.set_xlabel("Frequency [GHz]")
        ax1.set_title(f"Frequency vs Magnitude")
        ax2.set_title(f"Frequency vs Phase")
    
    # magn and phase
    ax1.plot(new_freqs, magn_dB, marker=marker, linestyle=linestyle, color=magn_color, markersize=4, label=magn_label, alpha=alpha)
    ax2.plot(new_freqs, phase_rad, marker=marker,  linestyle=linestyle, color=phase_color, markersize=4, label=phase_label, alpha=alpha)
    
    ax1.legend()
    ax1.set_ylabel("S21 [dB]")
    ax2.set_ylabel("Phase [Rad]")
    
    ax3.plot(real, imag, 'o', markersize=6)
    
    if zero_lines is True:
        ax3.axhline(0, linestyle=':', linewidth=1, color='k')
        ax3.axvline(0, linestyle=':', linewidth=1, color='k')
        
    ax3.set_title("Real vs Imag")
    ax3.set_xlabel("Real")
    ax3.set_ylabel("Imag")
    ax3.set_aspect('equal', adjustable='box')
    
    fig.suptitle(plot_title, fontsize=18, x=0.25)
    fig.tight_layout()
    
    return fig, axes



def makeSummaryFigure():
    fig = plt.figure(layout='constrained')
    ax = fig.subplot_mosaic([['smith', 'mag'], ['smith', 'phase']])
    fig.parameterAnnotation = None

    ax['mag'].sharex(ax['phase'])
    ax['phase'].set_xlabel('frequency (GHz)')
    ax['mag'].tick_params(labelbottom = False)
    ax['mag'].set_aspect('auto')
    ax['phase'].set_aspect('auto')
    #fig.subplots_adjust(hspace=0)#overridden by using the layout=constrain option in plt.figure()
    return fig, ax

def smith(sdata, **kwargs):
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)

    rf.plotting.plot_smith(sdata, ax=ax['smith'], x_label=None, y_label=None, title='Smith Chart', **kwargs)
    return fig, ax

#TODO: add support for linear magnitude
def magnitude(fdata, sdata, **kwargs):
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)

    ax['mag'].plot(fdata, 20 * np.log10(np.abs(sdata)), **kwargs)
    ax['mag'].set_ylabel('Magnitude (dB)')
    return fig, ax

#TODO: add support for degrees
def phase(fdata, sdata, **kwargs):
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)

    ax['phase'].plot(fdata, np.unwrap(np.angle(sdata)), **kwargs)
    ax['phase'].set_ylabel('phase (rad)')
    return fig, ax

def summaryPlot(fdata, sdata, **kwargs):
    '''
    This function combines plotres.smith(), .magnitude(), and .phase() functionality, passing **kwargs to
    the relevant matplotlib function
    '''
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)

    rf.plotting.plot_smith(sdata, ax=ax['smith'], x_label=None, y_label=None, title='Smith Chart', **kwargs)
    ax['mag'].plot(fdata, 20*np.log10(np.abs(sdata)), **kwargs)
    ax['mag'].set_ylabel('Magnitude (dB)')
    ax['phase'].plot(fdata, np.unwrap(np.angle(sdata)), **kwargs)
    ax['phase'].set_ylabel('phase (rad)')
    return fig, ax

def annotate(annotation_text: str):
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)

    if fig.parameterAnnotation == None:
        fig.parameterAnnotation = ax['smith'].annotate(str(annotation_text), (-1, -1.2), annotation_clip=False)
    else:
        text = fig.parameterAnnotation.get_text()
        text = text + str(annotation_text)
        fig.parameterAnnotation.set_text(text)

        x_pos, y_pos = fig.parameterAnnotation.get_position()
        fig.parameterAnnotation.set_position((x_pos, y_pos - 0.125))
        # TODO: query the font height & line spacing for the y-position adjustment

def annotateParam(param):
    fig = plt.gcf()
    ax_list = fig.get_axes()
    ax = AxesListToDict(ax_list)
    val = param.value
    stderr = param.stderr
    val, stderr = round_measured_value(val, stderr)

    #TODO: add a dictionary to convert parameter names to LaTeX symbols
    if fig.parameterAnnotation == None:
        fig.parameterAnnotation = ax['smith'].annotate(f'{param.name}= {val} +/- {stderr}', (-1,-1.2), annotation_clip=False)
    else:
        text = fig.parameterAnnotation.get_text()
        text = text+str('\n'+f'{param.name}= {val} +/- {stderr}')
        fig.parameterAnnotation.set_text(text)

        x_pos, y_pos = fig.parameterAnnotation.get_position()
        fig.parameterAnnotation.set_position((x_pos, y_pos-0.125))
        #TODO: query the font height & line spacing for the y-position adjustment


def displayAllParams(parameters):
    for key in parameters:
        annotateParam(parameters[key])

def AxesListToDict(ax_list):
    '''
    utility function to convert a list of matplotlib axes to a dictionary of them indexed by their label
    '''
    ax_dict = dict()
    for n in range(len(ax_list)):
        ax_dict.update({ax_list[n]._label: ax_list[n]})
    return ax_dict

#TODO: more careful verification of this function -- Google's AI gave it to me quicker than stackexchange
def round_measured_value(value, stdev):
    '''
    Rounding for measured quantities
    Two significant figures for the error
    value rounded to line up with first digit in the error
    '''
    place = int(np.floor(np.log10(stdev)))
    rounded_value = round(value, -place)
    rounded_err = round(stdev, -(place-1))
    return rounded_value, rounded_err

#TODO: display resonant & off-resonant points

