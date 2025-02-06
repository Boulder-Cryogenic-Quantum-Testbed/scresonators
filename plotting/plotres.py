import matplotlib.pyplot as plt
import skrf as rf
import numpy as np

#It might make sense to package all this into a subclass of matplotlib.Figure
#TODO: need to improve label placement when specified by user

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

