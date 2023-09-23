# -*- coding: utf-8 -*-
"""
Plotting functions
"""

import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import griddata

# Color blind friendly color scheme
plt.style.use('tableau-colorblind10')

class MPLPlotWrapper(object):
    """
    Class that wraps the matplotlib.pyplot with simpler, built-in functions
    for standard plotting and formatting commands
    """
    def __init__(self, *args, **kwargs):
        """
        Class constructor sets the default operations for the class
        """
        # Initialize the fontsizes and the figure, axes class members
        self.fsize          = 20
        self.tight_layout   = True
        self.leg            = None
        self.is_leg_outside = True
        self._xlabel        = ''
        self._ylabel        = ''
        self._xlim          = None
        self._ylim          = None
        self._xscale        = None
        self._yscale        = None
        self.plot           = None

        # Dimensions of the subplots
        self.xdim = 1
        self.ydim = 1

        # Update the arguments and keyword arguments
        self.__dict__.update(locals())
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.init_subplots()

        # Set the cyclers
        self.get_set_linestyle_cycler()
        self.get_set_alpha_color_cycler()
        self.get_set_marker_cycler()

    """
    Class properties
    """
    @property
    def xlabel(self):
        return self._xlabel
    @property
    def ylabel(self):
        return self._ylabel
    @property
    def xlim(self):
        return self._xlim
    @property
    def ylim(self):
        return self._ylim
    @property
    def xscale(self):
        return self._xscale
    @property
    def yscale(self):
        return self._yscale

    """
    Property deleters
    """
    @xlabel.deleter
    def xlabel(self):
        del self._xlabel
    @ylabel.deleter
    def ylabel(self):
        del self._ylabel
    @xscale.deleter
    def xlim(self):
        del self._xlim
    @yscale.deleter
    def yscale(self):
        del self._yscale
    @xscale.deleter
    def xscale(self):
        del self._xscale
    @ylim.deleter
    def ylim(self):
        del self._ylim

    """
    Property setters
    """
    @xlabel.setter
    def xlabel(self, xstr, fsize=None):
        ffsize = fsize if fsize is not None else self.fsize
        # Check the dimensions of the subplot
        if self.xdim > 1 or self.ydim > 1:
            for axx in self.ax.flat:
                axx.set(xlabel=xstr)
		    
            for axx in self.ax.flat:
                axx.label_outer() 
        else:
            self.ax.set_xlabel(xstr, fontsize=ffsize)

        self._xlabel = xstr
    @ylabel.setter
    def ylabel(self, ystr, fsize=None):
        ffsize = fsize if fsize is not None else self.fsize
        # Check the dimensions of the subplot
        if self.xdim > 1 or self.ydim > 1:
            for axx in self.ax.flat:
                axx.set(ylabel=ystr)
            for axx in self.ax.flat:
                axx.label_outer() 
        else:
            self.ax.set_ylabel(ystr, fontsize=ffsize)
            self._ylabel = ystr
    @xlim.setter
    def xlim(self, xlims=None):
        if self.xdim > 1 or self.ydim > 1:
            pass
        else:
            if xlims is not None:
                self.ax.set_xlim(xlims)
                self._xlim = xlims
    @ylim.setter
    def ylim(self, ylims=None):
        if self.xdim > 1 or self.ydim > 1:
            ylim_min = np.min([self.ax[i,j].get_ylim()[0] 
                for i in range(self.xdim) for j in range(self.ydim)])
            ylim_max = np.max([self.ax[i,j].get_ylim()[1] 
                for i in range(self.xdim) for j in range(self.ydim)])
            plt.setp(self.ax, ylim=[ylim_min, ylim_max])
        else:
            if ylims is not None:
                self.ax.set_ylim(ylims)
                self._ylim = ylims
    @xscale.setter
    def xscale(self, xscales=None):
        if self.xdim > 1 or self.ydim > 1:
            pass
        else:
            if xscales is not None:
                self.ax.set_xscale(xscales)
                self._xscale = xscales
    @yscale.setter
    def yscale(self, yscales=None):
        if self.xdim > 1 or self.ydim > 1:
            pass
        else:
            if yscales is not None:
                self.ax.set_yscale(yscales)
                self._yscale = yscales

    def close(self, op='all'):
        """
        Closes figures
        """
        plt.close(op)

    def init_subplots(self):
        """
        Returns a figure and axes object with the correct size fonts
        """
        # Get the figure, axes objects
        self.fig, self.ax = plt.subplots(self.xdim, self.ydim,
                tight_layout=self.tight_layout)
    
        # Set the ticks on all edges
        if self.xdim > 1 or self.ydim > 1:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    self.ax[i, j].tick_params(bottom=True, top=True, left=True,
                            right=True)
                    self.ax[i, j].tick_params(labelbottom=True, labeltop=False,
                            labelleft=True, labelright=False)
        else:
            self.ax.tick_params(bottom=True, top=True, left=True,
                    right=True)
            self.ax.tick_params(labelbottom=True, labeltop=False,
                    labelleft=True, labelright=False)
    
        # Set the tick label sizes
        self.set_axes_fonts()
        if self.xdim > 1 or self.ydim > 1:
            self.plot = np.array([self.ax[i,j].plot for i in range(self.xdim)
                                for j in range(self.ydim)])
        else:
            self.plot = self.ax.plot


    def get_set_linestyle_cycler(self):
        """
        Returns a linestyle cycler for plotting
        """
    
        # Different types of dashing styles
        self.linestyle_cycle = [
         (0, (1, 10)),
         (0, (1, 1)),
         (0, (1, 1)),
         (0, (5, 10)),
         (0, (5, 5)),
         (0, (5, 1)),
         (0, (3, 10, 1, 10)),
         (0, (3, 5, 1, 5)),
         (0, (3, 1, 1, 1)),
         (0, (3, 5, 1, 5, 1, 5)),
         (0, (3, 10, 1, 10, 1, 10)),
         (0, (3, 1, 1, 1, 1, 1))]
    
        return self.linestyle_cycle
    
     
    def get_set_alpha_color_cycler(self, alpha=0.5):
        """
        Returns color_cycler default with transparency fraction set to alpha
        """
    
        # Get the color cycler as a hex
        color_cycle_hex = plt.rcParams['axes.prop_cycle'].by_key()['color']
        hex2rgb = lambda hx: [int(hx[0:2],16)/256., \
                              int(hx[2:4],16)/256., \
                              int(hx[4:6],16)/256.]
        color_cycle_rgb = [hex2rgb(cc[1:]) for cc in color_cycle_hex]

        self.alpha_color_cycler = [(*cc, alpha) for cc in color_cycle_rgb]
    
        return self.alpha_color_cycler
    
    
    def get_set_marker_cycler(self):
        """
        Returns a marker style cycler for plotting
        """
    
        # Different marker icons
        self.marker_cycle = ['o', 's', 'D', 'x', 'v', '^', '*', '>', '<', 'p']
    
        return self.marker_cycle


    def set_str_axes_labels(self, axis='x'):
        """
        Sets the axes labels for `axis` to the strings in strs list
        """
        # Set the current axes to ax
        plt.sca(self.ax)
    
        # Select the appropriate axis to apply the labels
        if axis == 'x':
            plt.xticks(range(len(strs)), strs, fontsize=self.fsize)
        elif axis == 'y':
            plt.yticks(range(len(strs)), strs, fontsize=self.fsize)
        else:
            raise KeyError(f'axis ({axis}) invalid.')
    
    def set_axes_fonts(self, ax=None):
        """
        Set axes font sizes because it should be abstracted away
        """
        if ax is not None:
            for tick in ax.get_xticklabels():
                tick.set_fontsize(self.fsize)
            for tick in ax.get_yticklabels():
                tick.set_fontsize(self.fsize)
        else:
            if self.xdim > 1 or self.ydim > 1:
                for i in range(self.xdim):
                    for j in range(self.ydim):
                        for tick in self.ax[i, j].get_xticklabels():
                            tick.set_fontsize(self.fsize)
                        for tick in self.ax[i, j].get_yticklabels():
                            tick.set_fontsize(self.fsize)
            else:
                for tick in self.ax.get_xticklabels():
                    tick.set_fontsize(self.fsize)
                for tick in self.ax.get_yticklabels():
                    tick.set_fontsize(self.fsize)
    
    def set_axes_ticks(self, tk, ax=None, axis='x'):
        """
        Set the values displayed on the ticks
        """
        # Set the x ticks, otherwise, y ticks
        if ax is not None:
            if axis == 'x':
                ax.set_xticklabels(tk)
            elif axis == 'y':
                ax.set_yticklabels(tk)
            else:
                raise ValueError('(%s) axis is invalid.')
        else:

            if self.xdim > 1 or self.ydim > 1:
                for i in range(self.xdim):
                    for j in range(self.ydim):
                        if axis == 'x':
                            self.ax[i, j].set_xticklabels(tk)
                        elif axis == 'y':
                            self.ax[i, j].set_yticklabels(tk)
                        else:
                            raise ValueError('(%s) axis is invalid.')
            else:
                if axis == 'x':
                    self.ax.set_xticklabels(tk)
                elif axis == 'y':
                    self.ax.set_yticklabels(tk)
                else:
                    raise ValueError('(%s) axis is invalid.')
    
    def set_xaxis_rot(self, angle=45):
        """
        Rotate the x-axis labels
        """
        if self.xdim > 1 or self.ydim > 1:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    for tick in self.ax[i, j].get_xticklabels():
                        tick.set_rotation(angle)
        else:
            for tick in self.ax.get_xticklabels():
                tick.set_rotation(angle)

    def set_yaxis_rot(self, angle=45):
        """
        Rotate the y-axis labels
        """
        if self.xdim > 1 or self.ydim > 1:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    for tick in self.ax[i, j].get_yticklabels():
                        tick.set_rotation(angle)
        else:
            for tick in self.ax.get_yticklabels():
                tick.set_rotation(angle)


    def set_axes_num_format(self, fmt, axis='x'):
        """
        Sets the number format for the x and y axes in a 1D plot
        """
        if axis == 'x':
            self.ax.xaxis.set_major_formatter(
                    mpl.ticker.StrMethodFormatter(fmt))
        elif axis == 'y':
            self.ax.yaxis.set_major_formatter(
                    mpl.ticker.StrMethodFormatter(fmt))
        else:
            raise KeyError(f'axis {axis} not recognized.')
    

    def set_leg_outside(self, lsize=None):
        """
        Sets the legend location outside
        """
        # Set the legend fontsize to the user input or fsize
        fsize = self.fsize if lsize is None else lsize
        
        # Shrink current axis by 20%
        if self.xdim > 1 or self.ydim > 1:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    box = self.ax[i, j].get_position()
                    self.ax[i, j].set_position([box.x0, box.y0, box.width * 0.8,
                        box.height])
                    
                    # Put a legend to the right of the current axis
                    hdls, legs = self.ax[i, j].get_legend_handles_labels()
                    self.leg = self.ax[i, j].legend(hdls, legs,
                    loc='center left', bbox_to_anchor=(1, 0.5), fontsize=fsize,
                            framealpha=0.)
        else:

            box = self.ax.get_position()
            self.ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            
            # Put a legend to the right of the current axis
            hdls, legs = self.ax.get_legend_handles_labels()
            self.leg = self.ax.legend(hdls, legs, loc='center left',
                    bbox_to_anchor=(1, 0.5), fontsize=fsize, framealpha=0.)

        # Update the legend state
        self.is_leg_outside = True

    def plot_2d_cmap(self, x, y, z, fname, xstr='', ystr='', zstr='',
            tstr='', cbar_str='', cmap_str=cm.inferno, norm_type='linear',
            xyscales = {'x' : 'linear', 'y' : 'linear'},
            plot_option='imshow', zlim=None):
        """
        Plots a 2D heat map given data as x, y, z[2D]
        """
        # Get the dimensions and colormap
        NM = get_largest_factors(z.size)
        cmap = mpl.cm.get_cmap(cmap_str)

        # Plot with tricontourf interpolation
        if plot_option == 'tricontourf':
            # Normalize the data
            if norm_type == 'linear':
                norm = mpl.colors.Normalize(z.min(), z.max())
            elif norm_type == 'log':
                norm = mpl.colors.LogNorm(z.min(), z.max())
            else:
                raise ValueError(f'norm_type ({norm_type}) not supported.')

            # Overwrite the norm with an external reference
            if zlim is not None:
                norm = mpl.colors.Normalize(min(zlim), max(zlim))

            # Call the p-color mesh and set the labels
            plt1 = self.ax.tricontourf(x, y, z, norm=norm, cmap=cmap,
                    levels=100, extend='both')
            # This is the fix for the white lines between contour levels
            for c in plt1.collections:
                c.set_edgecolor('face')

            # plt1 = self.ax.scatter(x, y, c=z, cmap=cmap)

        elif plot_option == 'imshow':
            # Scatter plot with z-values as intensities
            z = z.reshape([max(NM), min(NM)])
            plt1 = self.ax.imshow(z,
                    extent=[x.min(), x.max(), y.min(), y.max()], cmap=cmap,
                    aspect='auto')

        elif plot_option == 'scatter':
            # Scatter plot with z-values as intensities
            plt1 = self.ax.scatter(x, y, c=z, cmap=cmap)

        else:
            raise ValueError(f'plot_option ({plot_option}) not recognized.')

        # Axes labels
        self.ax.set_xlabel(xstr, fontsize=self.fsize)
        self.ax.set_ylabel(ystr, fontsize=self.fsize)
        self.ax.set_title(tstr, fontsize=self.fsize)

        # Set the scales of the x- and y-axes
        self.ax.set_xscale(xyscales['x'])
        self.ax.set_yscale(xyscales['y'])

        # Rotate the axis labels
        self.set_xaxis_rot(angle=45.)

        # Set the colorbar
        cbar = self.fig.colorbar(plt1, ax=self.ax, format='%.0f')
        cbar.ax.set_title(cbar_str, fontsize=self.fsize, x=-0.1, y=1.05)
        cbar.ax.tick_params(labelsize=self.fsize)
        # cbar.set_ticks(np.arange(min(zlim), max(zlim), 5))

        # Write the figure to file
        self.write_fig_to_file(fname)

    def set_leg_hdls_lbs(self, lsize=None, loc='best'):
        """
        Set the legend handles and labels
        """
        # Set the legend fontsize to the user input or fsize
        fsize = self.fsize if lsize is None else lsize
        if self.xdim > 1 or self.ydim > 1:
            for i in range(self.xdim):
                for j in range(self.ydim):
                    hdl, leg = self.ax[i, j].get_legend_handles_labels()
                    self.ax[i, j].legend(hdl, leg, loc=loc, fontsize=fsize,
                                         framealpha=0.)
        else:
            hdls, legs = self.ax.get_legend_handles_labels()
            self.leg = self.ax.legend(hdls, legs, loc=loc, fontsize=fsize,
                                      framealpha=0.)

    def write_fig_to_file(self, fname):
        """
        Writes a figure object to file, sets legends accordingly
        """
        # Check for no legends
        format = fname.split('.')[-1]
        if self.leg is None:
            self.fig.savefig(fname, format=format, transparent=True)
        
        # Otherwise save with legends
        else:
            ## Check for setting legend outside
            if self.is_leg_outside:
                self.fig.savefig(fname, format=format,
                        bbox_extra_artists=(self.leg, ), bbox_inches='tight',
                        transparent=True)
            else:
                ### Check for the ax object to set the legends
                self.fig.savefig(fname, format=format, transparent=True)

        print(f'{fname} written to file.')
