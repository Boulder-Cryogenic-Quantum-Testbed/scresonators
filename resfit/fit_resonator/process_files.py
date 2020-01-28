# -*- coding: utf-8 -*-
"""
Created on Fri May 11 19:37:41 2018

@author: hung93
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import inflect
import os
import fnmatch
from datetime import datetime
from matplotlib.gridspec import GridSpec
import re
import matplotlib.pylab as pylab
from .fit_S_data import fit_resonator,Cavity_DCM,Cavity_inverse
from .resonator import Resonator,FitMethod
import ast
params = {'legend.fontsize': 18,
          'figure.figsize': (10, 8),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize':18,
         'ytick.labelsize':18,
         'lines.markersize' : 1,
         'font.size': 15.0 }
pylab.rcParams.update(params)

np.set_printoptions(precision=4,suppress=True)

p = inflect.engine() # search ordinal

#############################################
def List_resonators(dic,delay):
    fname = []
    for file in os.listdir(dic):
        if fnmatch.fnmatch(file,'*Pow*'):
            if fnmatch.fnmatch(file,'*[!jpg]'):
                if fnmatch.fnmatch(file,'*[!xlsx]'):
                    fname.append(file)
    Resonators_array = []
    for file in fname:
        filepath = dic+'\\' + file
        measure_detail = re.findall('[+-]?\d+',file)
        time = measure_detail[1]
        date = measure_detail[0]
        date_time = datetime.strptime(date+time,"%Y%m%d%H%M%S")
        power = measure_detail[2]
        name = file.split(".txt")[0]
        if fnmatch.fnmatch(file,'*Bias*'):
            bias = file.split('Bias')[1].split('.txt')[0]
        else:
            bias = None
        data = np.genfromtxt(filepath)
        xdata = data.T[0]/10**9
        y1data = data.T[1]
        y2data = data.T[2]
        ## check if VNA takes data, sometimes it doesnot
        if (y1data == 0).all():
            os.remove(filepath)
            break
        Resonators_array.append(resonator(\
        xdata, y1data, y2data, power, delay, name = name, date = date_time,bias = bias))
    return Resonators_array

#################################################
def MultiFit(dic,list_resonators,method,fit = 'coarse'):
    path = dic+'\\Fig_extract factor_'+str(method.extract_factor)+'_weighing_'+str(method.MC_weight)
    if not os.path.exists(path):
        os.makedirs(path)
    i = 0
    Params_array = []
    init_array = []
    chi_array = []
    manual_init = method.manual_init
    for k in list_resonators:
        print("Current fitting index " + str(i))
        params,fig,chi,init = fit_resonator(k, method)
        filepath = path + '\\'+method.method+'_'+k.name+".jpg"
        k.load_params(method.method,params,chi)
        fig.savefig(filepath)
#        print(k)
#        print(init[1:5])
#        print('\n')
#        method.vary = [False,True,True,False,True,False]
#        params,fig,chi,init = Fit_Resonator(k,method)
#        filepath = dic +'\\Fig\\'+k.name+'_'+method.method+"1.jpg"
#        fig.savefig(filepath)
        Params_array.append(params)
        init_array.append(init)
        chi_array.append(chi)
        plt.close()
        i = i+1
    Params_array = np.array(Params_array)
    chi_array = np.array(chi_array)
    bad_fit, = np.where(chi_array > 2*np.average(chi_array))
    print(bad_fit)
    for k in bad_fit:
        if (k-1) in bad_fit:
            method.manual_init= [k for k in Params_array[k-2]]
        if k != 0:
            method.manual_init= [k for k in Params_array[k-1]]
        elif k== 0:
            method.manual_init= [k for k in Params_array[1]]
        print(method.manual_init)
        params,fig,chi,init = fit_resonator(list_resonators[k], method)
        if chi < chi_array[k]:
            print(list_resonators[k].name+' changes the fitting\n chi: ' + str(chi_array[k])+' to '+str(chi))
            filepath = path + '\\' +method.method+'_'+list_resonators[k].name+"_second.jpg"
            list_resonators[k].reload_params(method.method,params,chi)
            fig.savefig(filepath)
            Params_array[k] = params
            init_array[k] = init
            plt.close()
    if fit == 'coarse':
        return Params_array,np.array(init_array)
    elif fit == 'fine':
        factor = method.extract_factor
        method.manual_init = manual_init
        i = 0
        for k in list_resonators:
            if method.method == 'DCM':
                df = k.DCMparams.fc*factor/k.DCMparams.Q
                fc = k.DCMparams.fc
            elif method.method == 'INV':
                df = k.INVparams.fc*factor/k.INVparams.Q
                fc = k.INVparams.fc
            method.extract_factor = [fc-df/2,fc + df/2]
            params,fig,chi,init = fit_resonator(k, method)
            print("Current finer fitting index " + str(i))
            filepath = path  +'\\Fine_'+method.method+'_'+k.name+".jpg"
            k.reload_params(method.method,params,chi)
            fig.savefig(filepath)
            plt.close()
            i = i+1
        return Params_array,np.array(init_array)
########################################################################
def Result_dataframe(dic,list_resonators,Method = None):
    df1 = pd.DataFrame()
    name = [s.name for s in list_resonators]
    df1['name'] = pd.Series(name)
    span = [s.span for s in list_resonators]
    df1['span'] = pd.Series(span)
    center_freq = [s.center_freq for s in list_resonators]
    df1['center_freq'] = pd.Series(center_freq)
    temp = [s.temp for s in list_resonators]
    df1['temp'] = pd.Series(temp)
    bias = [s.bias for s in list_resonators]
    df1['bias'] = pd.Series(bias)
    method = [s.method for s in list_resonators]
    df1['method'] = pd.Series(method)
    corrected_power= [s.corrected_power for s in list_resonators]
    df1['corrected_power'] = pd.Series(corrected_power)
    S21= [s.S21 for s in list_resonators]
    df1['S21'] = pd.Series(S21)
    compare= [s.compare for s in list_resonators]
    df1['compare'] = pd.Series(compare)
    if 'DCM' in list_resonators[0].method:
        DCM_fc = [s.DCMparams.fc for s in list_resonators]
        df1['DCM_fc'] = pd.Series(DCM_fc)
        DCM_Qe = [s.DCMparams.Qe for s in list_resonators]
        df1['DCM_Qe'] = pd.Series(DCM_Qe)
        DCM_Q = [s.DCMparams.Q for s in list_resonators]
        df1['DCM_Q'] = pd.Series(DCM_Q)
        DCM_ReQe = [s.DCMparams.ReQe for s in list_resonators]
        df1['DCM_ReQe'] = pd.Series(DCM_ReQe)
        DCM_Qi = [s.DCMparams.Qi for s in list_resonators]
        df1['DCM_Qi'] = pd.Series(DCM_Qi)
        DCM_chi = [s.DCMparams.chi for s in list_resonators]
        df1['DCM_chi'] = pd.Series(DCM_chi)
        DCM_phi = [s.DCMparams.phi for s in list_resonators]
        df1['DCM_phi'] = pd.Series(DCM_phi)
        DCM_num_photon = [s.DCMparams.num_photon for s in list_resonators]
        df1['DCM_num_photon'] = pd.Series(DCM_num_photon)
        DCM_all = [s.DCMparams.all for s in list_resonators]
        df1['DCM_all'] = pd.Series(DCM_all)
    if 'INV' in list_resonators[0].method:
        INV_fc = [s.INVparams.fc for s in list_resonators]
        df1['INV_fc'] = pd.Series(INV_fc)
        INV_Q = [s.INVparams.Q for s in list_resonators]
        df1['INV_Q'] = pd.Series(INV_Q)
        INV_Qe = [s.INVparams.Qe for s in list_resonators]
        df1['INV_Qe'] = pd.Series(INV_Qe)
        INV_Qi = [s.INVparams.Qi for s in list_resonators]
        df1['INV_Qi'] = pd.Series(INV_Qi)
        INV_chi = [s.INVparams.chi for s in list_resonators]
        df1['INV_chi'] = pd.Series(INV_chi)
        INV_phi = [s.INVparams.phi for s in list_resonators]
        df1['INV_phi'] = pd.Series(INV_phi)
        INV_num_photon = [s.INVparams.num_photon for s in list_resonators]
        df1['INV_num_photon'] = pd.Series(INV_num_photon)
        INV_all = [s.INVparams.all for s in list_resonators]
        df1['INV_all'] = pd.Series(INV_all)
    df1= df1.sort_values(['corrected_power','bias'])
    df1 = df1.reset_index(drop = True)
    name = dic+'/dataframe_params_extract factor_'+str(Method.extract_factor)+'_weighing_'+str(Method.MC_weight)+'.pkl'
    df1.to_pickle(name)
    name = dic+'\\Fig_extract factor_'+str(Method.extract_factor)+'_weighing_'+str(Method.MC_weight)+'/dataframe_params_extract factor_'+str(Method.extract_factor)+'_weighing_'+str(Method.MC_weight)+'.pkl'
    df1.to_pickle(name)
    name = dic+'/dataframe_params_extract factor_'+str(Method.extract_factor)+'_weighing_'+str(Method.MC_weight)+'.xlsx'
    writer = pd.ExcelWriter(name)
    df1.to_excel(writer,'Sheet1')
    writer.save()

    return df1

########################################################################
                    #      Set Temperature DataFram       ##
def temp_log(filepath):
    data = np.loadtxt(dic_temp+'\\'+fr,dtype='str')
    xx = data.T[0].tolist()
    yy = data.T[1].tolist()
    date_time = [datetime.strptime(a+' ' +b,"%Y-%m-%d %H:%M:%S") for a, b in zip(xx, yy)]
    R_3K = data.T[2]
    R_700mK = data.T[3]
    R_50mK = data.T[4]
    R_10mK = data.T[5]

    temp_3K = data.T[10]
    temp_700mK = data.T[11]
    temp_50mK = data.T[12]
    temp_10mK = data.T[13]
    temp_RT = np.array(data.T[14],dtype = float)
    temp_RT[temp_RT < 0] = 0

    df_temp = pd.DataFrame()
    df_temp['Date/Time'] = pd.Series(date_time)
    df_temp['temp_3K'] = pd.Series(temp_3K)
    df_temp['temp_700mK'] = pd.Series(temp_700mK)
    df_temp['temp_50mK'] = pd.Series(temp_50mK)
    df_temp['temp_10mK'] = pd.Series(temp_10mK)
    df_temp['temp_RT'] = pd.Series(temp_RT/1000) # mK to K
    df_temp['R_3K'] = pd.Series(R_3K)
    df_temp['R_700mK'] = pd.Series(R_700mK)
    df_temp['R_50mK'] = pd.Series(R_50mK)
    df_temp['R_10mK'] = pd.Series(R_10mK)

    # Save DataFrame
    df_temp.to_pickle(dic_temp+'/'+fr.split('.dat')[0]+'.pkl')

    return df_temp
#
#####################################################################
#                ## Add Temperauture to Original DataFrame   ##
#df_temp = pd.read_pickle(dic_temp+'\\'+fr)
def add_Res_temp(df_temp,resonator):
    near_date = resonator.date
    nearest_date = nearest(df_temp['Date/Time'],near_date)
    temp_10mK = pd.to_numeric(df_temp[df_temp['Date/Time'] ==nearest_date]['temp_10mK'].tolist())
    resonator.add_temp(temp_10mK)
    return None
########################################################
def add_list_Res_temp(filepath,list_resonators):
    df_temp = temp_log(filepath)

    for k in range(len(list_resonators)):                     # add temperature to list of resonators
        add_Res_temp(df_temp,list_resonators[k])
    return None
########################################################
def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

########################################################
def convert_params(Method,params):
    if Method =='DCM':
        Qe = params[2]/np.cos(params[4])
        Qi = params[1]*Qe/(Qe-params[1])
    elif Method == 'INV':
        Qe = params[2]
        Qi = params[1]
    elif Method == 'CPZM':
        Qe = params[2]
        Qi = params[1]
#    print('Qi, Qe = ' + str(Qi) + ', '+str(Qe))
    return Qe,Qi

###########################################################
def Plot_sweep_S21(x,y,z,figurename,plotrange = [0,0],xlabel = 'Power (dbm)',different_length = False,cmap = None,Log10= True):
    ## plot sweep temperature/bias/time vs frequency vs S21 (db)
    if cmap ==None:
        cmap = plt.get_cmap('jet')
    plotrange = [min(plotrange),max(plotrange)]
    if all([isinstance(xx, float) for xx in x]) ==False:
        x = np.asarray(x).astype(np.float)
#    else:
#        x = x.as_matrix()
    y = y
    z = np.abs(np.array(z))

    ## adjust measurement with different sweep points
    if different_length == True:
        new_z = []
        max_total_pts = 0
        for yy in y:
            new_total_pts = len(yy)
            if new_total_pts > max_total_pts:
                max_total_pts = new_total_pts

        for i in range(len(y)):
            y1 = y[i]
            new_y1 = np.linspace(y1.min(),y1.max(),max_total_pts)
            z1 = np.abs(z[i])
            new_z1 = sp.interpolate.interp1d(y1, z1, kind='linear')(new_y1)
            new_z.append(new_z1)
        y = new_y1
        z = 20*np.log10(new_z)
    if different_length != True:
        z = 20*np.log10(z)
    if Log10 != True:
        z = 10**(z/20)

    plt.close(figurename)
    fig = plt.figure(figurename,figsize = [12,8])

    xbins = len(x)
    ybins = len(y)
    xi, yi = np.mgrid[x.min():x.max():xbins*1j, y.min():y.max():ybins*1j]
    if plotrange != [0,0]:
        plt.pcolormesh(xi,yi,z,cmap=cmap,vmin=plotrange[0], vmax=plotrange[1])
    else:
        plt.pcolormesh(xi,yi,z,cmap = cmap)
    plt.xlabel(xlabel,fontsize=16)
    plt.ylabel('frequency (GHz)',fontsize=16)
    cb = plt.colorbar()
    cb.set_label(r"$S_{21}$ $(db)$",fontsize=16)
    plt.show()
    return fig

##############################################################################
def convert_diff_method(method1,method2,params):
    if method2 == 'Diff':
        Qe,Qi = convert_params(method1,params)
        if method1 =='DCM':
            Qe_INV = params[2]
            Qi_INV = Qi/(1+Qe_INV/2/np.sin(params[4]))

            return [1/params[0],Qi_INV,Qe_INV,params[3],-params[4],-params[5]]
        elif method1 == 'INV':
            Qe_DCM = params[2]
            Qi_DCM = params[1]*(params[2]/2/np.sin(params[4])+1)
            Q_DCM = ( np.cos(params[4])/params[2]+1/params[1])**-1
            return [1/params[0],Q_DCM,Qe_DCM,params[3],-params[4],-params[5]]

    else:
        Qe,Qi = convert_params(method1,params)
        if method1 =='DCM' and method2 == 'INV':
            Qe_INV = params[2]
            Qi_INV = Qi/(1+np.sin(params[4])/Qe_INV/2)
            print(Qi,Qi_INV,Qe_INV)
            return [1/params[0],Qi_INV,Qe_INV,params[3],-params[4],-params[5]]
        elif method1 == 'INV' and method2 == 'DCM':
            Qe_DCM = params[2]
            Qi_DCM = params[1]*(params[2]/2/np.sin(params[4])+1)
            Q_DCM = ( np.cos(params[4])/params[2]+1/params[1])**-1
            return [1/params[0],Q_DCM,Qe_DCM,params[3],-params[4],-params[5]]

def read_method(dic):
    df = pd.read_excel(dic)
    i = 0
    for k in list(df.columns.values):
        df = df.reset_index()
        if k == 'DCM':
            i = i+1
            Method1 = FitMethod('DCM')
            delay = df[k][0]
            if pd.notna(df[k][1]):
                Method1.extract_factor = df[k][1]
            if pd.notna(df[k][2]):
                Method1.MC_iteration = df[k][2]
            if pd.notna(df[k][3]):
                Method1.MC_rounds = df[k][3]
            if pd.notna(df[k][4]):
                Method1.MC_weight = df[k][4]
            if pd.notna(df[k][5]):
                Method1.MC_weightvalue = df[k][5]
            if pd.notna(df[k][6]):
                Method1.MC_fix = df[k][6]
            if pd.notna(df[k][7]):
                Method1.MC_step_const = df[k][7]
            if pd.notna(df[k][8]):
                Method1.find_circle = df[k][8]
            if pd.notna(df[k][9]):
                Method1.vary =  df[k][9]
            if pd.notna(df[k][10]):
                Method1.manual_init = ast.literal_eval(df[k][10])

        if k == 'INV':
            i = i+1
            Method2 = FitMethod('INV')
            delay = df[k][0]
            if pd.notna(df[k][1]):
                Method2.extract_factor = df[k][1]
            if pd.notna(df[k][2]):
                Method2.MC_iteration = df[k][2]
            if pd.notna(df[k][3]):
                Method2.MC_rounds = df[k][3]
            if pd.notna(df[k][4]):
                Method2.MC_weight = df[k][4]
            if pd.notna(df[k][5]):
                Method2.MC_weightvalue = df[k][5]
            if pd.notna(df[k][6]):
                Method2.MC_fix = df[k][6]
            if pd.notna(df[k][7]):
                Method2.MC_step_const = df[k][7]
            if pd.notna(df[k][8]):
                Method2.find_circle = df[k][8]
            if pd.notna(df[k][9]):
                Method2.vary =  df[k][9]
            if pd.notna(df[k][10]):
                Method2.manual_init = ast.literal_eval(df[k][10])
    if i == 2:
        print('Two methods in file')
        return Method1,Method2,delay
    elif i == 1:
        if 'Method1' in locals():
            print('Create DCM fit in file')
            return Method1,delay
        elif 'Method2' in locals():
            print('Create INV fit in file')
            return Method2,delay
        else:
            print('false')
def save_method():
    return None
def Plot_iDCM_INV(dic,list_resonators,method,base = "INV"):
    params = {'legend.fontsize': 18,
          'figure.figsize': (10, 8),
         'axes.labelsize': 18,
         'axes.titlesize':18,
         'xtick.labelsize':18,
         'ytick.labelsize':18,
         'lines.markersize' : 3,
         'lines.linewidth' : 5,
         'font.size': 15.0 }
    pylab.rcParams.update(params)
    fig_path = dic+'\\Fig_extract factor_'+str(method.extract_factor)+'_weighing_'+str(method.MC_weight)+'\\Compare_Fig'
    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    i = 0
    for k in list_resonators:
        if method.MC_weight == 'yes':
            DCM_label = 'iDCM fit'
            columns = ('iDCM', 'INV')
        elif method.MC_weight == 'no':
            DCM_label = 'DCM fit'
            columns = ('DCM', 'INV')
        factor = method.extract_factor
        if method.method == 'DCM':
            df = k.DCMparams.fc*factor/k.DCMparams.Q
            fc = k.DCMparams.fc
        elif method.method == 'INV':
            df = k.INVparams.fc*factor/k.INVparams.Q
            fc = k.INVparams.fc
        extract_factor = [fc-df/2,fc + df/2]
        print(k.freq[0],k.freq[-1])
        print(extract_factor)
        i = i+1
        print(i)
        DCM_params = k.DCMparams.all
        INV_params = k.INVparams.all
        x = k.freq
        points = int(15/factor)
        x_fit2 = np.append(np.linspace(x[0],extract_factor[0],points),np.linspace(extract_factor[1],x[-1],points))
        x_fit = np.linspace(extract_factor[0],extract_factor[1],5000)
        if base == "INV":
            y = 1/k.S21
            DCM_y_fit2 = 1/Cavity_DCM(x_fit2,*DCM_params)
            INV_y_fit2 = Cavity_inverse(x_fit2,*INV_params)
            DCM_y_fit = 1/Cavity_DCM(x_fit,*DCM_params)
            INV_y_fit = Cavity_inverse(x_fit,*INV_params)

        elif base == 'DCM':
            y = k.S21
            DCM_y_fit2 = Cavity_DCM(x_fit2,*DCM_params)
            INV_y_fit2 = 1/Cavity_inverse(x_fit2,*INV_params)
            DCM_y_fit = Cavity_DCM(x_fit,*DCM_params)
            INV_y_fit = 1/Cavity_inverse(x_fit,*INV_params)

        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3,3)
        ax1 = plt.subplot(gs[0,0]) ## real part
        ax2 = plt.subplot(gs[0,1]) ## imag part
        ax3 = plt.subplot(gs[0,2]) ## magnitude
        ax4 = plt.subplot(gs[1,2])
        ax = plt.subplot(gs[1:3,0:2]) ## IQ plot

        ax1.plot(x,np.real(y),'-',color = 'skyblue',label = 'raw')
        ax1.plot(x_fit2,np.real(DCM_y_fit2),'o',color = 'lightgreen')
        ax1.plot(x_fit2,np.real(INV_y_fit2),'o',color = 'lightcoral')
        ax1.plot(x_fit,np.real(DCM_y_fit),'r-',label = DCM_label)
        ax1.plot(x_fit,np.real(INV_y_fit),'g-',label = 'INV fit')
        ax1.set_xlim(left=x[0], right=x[-1])
        if base == "INV":
            ax1.set_title("real part $S_{21}^{-1}$")
            plt.ylabel('Re[$S_{21}^{-1}$]')
            plt.xlabel('frequency (GHz)')
        if base == "DCM":
            ax1.set_title("real part $S_{21}$")
            plt.ylabel('Re[$S_{21}$]')
            plt.xlabel('frequency (GHz)')

        ax2.plot(x,np.imag(y),'b-',label = 'raw')
        ax2.plot(x_fit2,np.imag(DCM_y_fit2),'ro')
        ax2.plot(x_fit2,np.imag(INV_y_fit2),'go')
        ax2.plot(x_fit,np.imag(DCM_y_fit),'r-',label = DCM_label)
        ax2.plot(x_fit,np.imag(INV_y_fit),'g-',label = 'INV fit')
        ax2.set_xlim(left=x[0], right=x[-1])
        if base == "INV":
            ax2.set_title("Imag part $S_{21}^{-1}$")
            plt.ylabel('Im[$S_{21}^{-1}$]')
            plt.xlabel('frequency (GHz)')
        if base == "DCM":
            ax2.set_title("Imag part $S_{21}$")
            plt.ylabel('Im[$S_{21}$]')
            plt.xlabel('frequency (GHz)')

        ax3.plot(x,np.abs(y),'b-',label = 'raw')
        ax3.plot(x_fit2,np.abs(DCM_y_fit2),'ro')
        ax3.plot(x_fit2,np.abs(INV_y_fit2),'go')
        ax3.plot(x_fit,np.abs(DCM_y_fit),'r-',label = DCM_label)
        ax3.plot(x_fit,np.abs(INV_y_fit),'g-',label = 'INV fit')
        ax3.set_xlim(left=x[0], right=x[-1])
        if base == "INV":
            ax3.set_title("Magnitude part $S_{21}^{-1}$")
            plt.ylabel('Abs[$S_{21}^{-1}$]')
            plt.xlabel('frequency (GHz)')
        if base == "DCM":
            ax3.set_title("Magnitude part $S_{21}$")
            plt.ylabel('Abs[$S_{21}$]')
            plt.xlabel('frequency (GHz)')

        params = {'legend.fontsize': 18,
              'figure.figsize': (10, 8),
             'axes.labelsize': 18,
             'axes.titlesize':18,
             'xtick.labelsize':18,
             'ytick.labelsize':18,
             'lines.markersize' : 7,
             'lines.linewidth' : 4,
             'font.size': 15.0 }
        pylab.rcParams.update(params)
        y = y[np.where(abs(y)<3500)]
        ax.plot(np.real(y),np.imag(y),'o',color = 'k',label = 'raw',markersize = 2)
        ax.plot(np.real(DCM_y_fit2),np.imag(DCM_y_fit2),'o',color = 'indianred',markersize = 3)
        ax.plot(np.real(INV_y_fit2),np.imag(INV_y_fit2),'o',color = 'lightgreen',markersize = 3)
        ax.plot(np.real(DCM_y_fit),np.imag(DCM_y_fit),'r-',label = DCM_label)
        ax.plot(np.real(INV_y_fit),np.imag(INV_y_fit),'g-',label = 'INV fit')
        ax.set_ylim(-100,6000)
        plt.axis('square')
        if base == "INV":
            plt.title("Real and Imag part $S_{21}^{-1}$")
            plt.ylabel('Im[$S_{21}^{-1}$]')
            plt.xlabel("Re[$S_{21}^{-1}$]")
        if base == "DCM":
            plt.title("Real and Imag part $S_{21}$")
            plt.ylabel('Im[$S_{21}$]')
            plt.xlabel("Re[$S_{21}$]")

        leg = plt.legend()
    # get the individual lines inside legend and set line width
        for line in leg.get_lines():
            line.set_linewidth(10)

        Qe = DCM_params[2]*np.exp(1j*DCM_params[4])
        Qi = (DCM_params[1]**-1-abs(np.real(Qe**-1)))**-1
        Q_INV = (INV_params[2]**-1 + INV_params[1]**-1)**-1
        rows = [r'1/Re[1/$Q_e$]',r'$Q_i$',r'$|Q_e|$','Q',r'$f_c$','$\phi  ( \degree)$']
        table = [['%1.2f'% (1/np.real(Qe**-1)),'%1.2f'% INV_params[2]],\
                  ['%1.2f'% Qi,'%1.2f'% INV_params[1]],\
                  ['%1.2f'% DCM_params[2],'%1.2f'% INV_params[2]],\
                  ['%1.2f'% DCM_params[1],'%1.2f'% Q_INV],\
                  ['%1.5f'% DCM_params[3],'%1.5f'% INV_params[3]],\
                  ['%1.2f'% (np.rad2deg(DCM_params[4])% 360),'%1.2f'% (np.rad2deg(-INV_params[4])% 360)]]
        ax4.axis('tight')
        ax4.axis('off')
        the_table = ax4.table(cellText=table,rowLabels=rows,
                      colLabels=columns,loc='center')
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(16)
        the_table.scale(1,1.5)
        plt.tight_layout()
        if base == "INV":
            fig.savefig(fig_path+'\\INV_Comapred'+k.name+'.jpg')
        if base == "DCM":
            fig.savefig(fig_path+'\\DCM_Comapred'+k.name+'.jpg')
        plt.close()
    return None
