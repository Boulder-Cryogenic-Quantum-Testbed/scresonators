# -*- encoding: utf-8 -*-	    
"""
User file for controlling the Janis and PNA instruments

    Make sure you login to the JetWay session on PuTTY
    with user: 'bco' and password: 'aish8Hu8'

"""
import sys

# Change this path
sys.path.append(r'C:\Users\Lehnert Lab\GitHub\bcqt-ctrl\temperature_control')
from janis_ctrl import measure_multiple_resonators
import numpy as np
import time


# Set the center frequencies (GHz), spans (MHz), delays(ns), powers
   
fcs = [4.7830068, 5.20607025, 5.60497675, 5.9866719,
        6.424396, 6.85450095, 7.2714644, 7.72134665]
spans = [0.5, 0.3, 0.5, 1.0,
        10.0, 0.5, 0.5, 0.5]
delays = [59.96, 59.91, 59.81, 59.83,
          59.84, 59.82, 59.79, 59.72]


# Change the sample name
sample_name = 'NWOXCTRL02'

'''
powers = np.linspace(-15, -35, 5)
measure_multiple_resonators(fcs, spans, delays, powers,
        ifbw=1., sparam='S21', npts=51,
        adaptive_averaging=False, sample_name=sample_name,
        runtime=1., cal_set = None, start_delay=0.,
        is_segmented=True, offresfraction=0.8, use_homophasal=None,
        Navg_init=None)
'''

powers = np.linspace(-40, -70, 7)
measure_multiple_resonators(fcs, spans, delays, powers,
       ifbw=1., sparam='S21', npts=51,
       adaptive_averaging=True, sample_name=sample_name,
       runtime=0.25, cal_set = None, start_delay=0.,
       is_segmented=True, offresfraction=0.8, use_homophasal=None,
       Navg_init=None)



powers = np.linspace(-75, -95, 5)
measure_multiple_resonators(fcs, spans, delays, powers,
       ifbw=1., sparam='S21', npts=51,
       adaptive_averaging=True, sample_name=sample_name,
       runtime=4., cal_set = None, start_delay=0.,
       is_segmented=True, offresfraction=0.8, use_homophasal=None,
       Navg_init=None)

