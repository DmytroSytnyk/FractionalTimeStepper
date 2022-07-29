import sys
import os
import matplotlib
import numpy as np
from math import isinf
import random
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def plot_stuff(model_setting,QoIs,alpha_val,time,save_at):
    
    fig = plt.figure()
    ax  = plt.subplot(111)
    
    for i in range(0,np.size(alpha_val)):
        print('Alpha= ',alpha_val[i])
        print('Mass= ', QoIs[i][0])
        ax.plot(time,QoIs[i][0], label=r'$\alpha = %.2f$' %alpha_val[i])
        plt.ylabel('Mass')
        plt.xlabel('Time')
        ax.legend()
        plt.savefig(save_at +'mass.png')

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(0,np.size(alpha_val)):
        print('Alpha= ',alpha_val[i])
        print("Roughness= ",QoIs[i][1])
        ax.plot(time,QoIs[i][1],label=r'$\alpha = %.2f$' %alpha_val[i])
        plt.ylabel('Roughness')
        plt.xlabel('Time')
        ax.legend()
        plt.savefig(save_at +'roughness.png')
        
        
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(0,np.size(alpha_val)):
        print('Alpha= ',alpha_val[i])
        print("Energy= ",QoIs[i][2])
        ax.plot(time,QoIs[i][2], label=r'$\alpha = %.2f$' %alpha_val[i])
        plt.ylabel('Energy')
        plt.xlabel('Time')
        ax.legend()
        plt.savefig(save_at +'energy.png')
        
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(0,np.size(alpha_val)):
        print('Alpha= ',alpha_val[i])
        print("Runtime= ",QoIs[i][3])
        runtime = np.delete(QoIs[i][3],0)
        numbers = np.arange(len(runtime))
        ax.plot(numbers,runtime, label=r'$\alpha = %.2f$' %alpha_val[i])
        plt.ylabel('Runtime')
        plt.xlabel('Memory')
        ax.legend()
        plt.savefig(save_at +'runtime.png')
        
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(0,np.size(alpha_val)):
        runtime = np.delete(QoIs[i][3],0)
        numbers = np.arange(len(runtime))
        interpol_x = np.linspace(0,len(runtime),2)
        interpol_y = np.interp(interpol_x,numbers,runtime)
        ax.plot(interpol_x,interpol_y, label=r'$\alpha = %.2f$' %alpha_val[i])
        plt.ylabel('Interpolated runtime')
        plt.xlabel('Memory')
        ax.legend()
        plt.savefig(save_at +'runtime_interpol.png')
        
    return


def plot_all(QoIs_left,QoIs_right,alpha_val,time,save_at):
    
    fig = plt.figure()
    ax  = plt.subplot(111)
    
    for i in range(0,np.size(alpha_val)):
        print('Alpha= ',alpha_val[i])
        print('Mass(L)= ', QoIs_left[i][0])
        print('Mass(R)= ', QoIs_right[i][0])
        ax.plot(time,QoIs_left[i][0], label=r'$\alpha = %.2f$ (Left)' %alpha_val[i])
        ax.plot(time,QoIs_right[i][0], label=r'$\alpha = %.2f$ (Right)' %alpha_val[i])
        plt.ylabel('Mass')
        plt.xlabel('Time')
        ax.legend()
        plt.savefig(save_at +'mass_all.png')

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(0,np.size(alpha_val)):
        print('Alpha= ',alpha_val[i])
        print("Roughness(L)= ",QoIs_left[i][1])
        print("Roughness(R)= ",QoIs_right[i][1])
        ax.plot(time,QoIs_left[i][1], label=r'$\alpha = %.2f$ (Left)' %alpha_val[i])
        ax.plot(time,QoIs_right[i][1], label=r'$\alpha = %.2f$ (Right)' %alpha_val[i])
        plt.ylabel('Roughness')
        plt.xlabel('Time')
        ax.legend()
        plt.savefig(save_at +'roughness_all.png')
        
        
    fig = plt.figure() 
    ax = plt.subplot(111)
    for i in range(0,np.size(alpha_val)):
        print('Alpha= ',alpha_val[i])
        print("Energy(L)= ",QoIs_left[i][2])
        print("Energy(R)= ",QoIs_right[i][2])
        ax.plot(time,QoIs_left[i][2], label=r'$\alpha = %.2f$ (Left)' %alpha_val[i])
        ax.plot(time,QoIs_right[i][2], label=r'$\alpha = %.2f$ (Right)' %alpha_val[i])
        plt.ylabel('Energy')
        plt.xlabel('Time')
        ax.legend()
        plt.savefig(save_at +'energy_all.png')


def pvd_set_nearest_t(reader, t):
    if (t < 0):
        # Use the first time point 
        reader.set_active_time_point(0)
    else:
        if isinf(float(t)):
            reader.set_active_time_point(reader.number_time_points-1)
        else:
            t_idx = np.abs(np.array(reader.time_values)-t).argmin()
            reader.set_active_time_point(t_idx)
    return reader.active_time_value

def render_pvd2file_2d_color(filename, plottime, **kwargs):
    import pyvista as pv
    pv.set_plot_theme('document')
    # Details are at https://docs.pyvista.org/api/readers/_autosummary/pyvista.PVDReader.html
    reader = pv.get_reader(filename)
    # Print available time values
    # print(reader.time_values)
    t = pvd_set_nearest_t(reader,plottime)
    # if not plottime:
        # # Use the first time point 
        # reader.set_active_time_point(0)
    # else:
        # import pdb; pdb.set_trace()    
        # if isinf(float(plottime)):
            # reader.set_active_time_point(reader.number_time_points-1)
        # else:
            # t_idx = np.abs(np.array(reader.time_values)-plottime).argmin()
            # reader.set_active_time_point(t_idx)
    print(f"Plotting data for t = {t:.3f}") 
    mesh = reader.read()[0]
    # Print available point data
    # print(mesh.point_data)  
    # mesh.plot(cpos='xy')
    plotter = pv.Plotter(off_screen=True)
    datatitle = kwargs.get('datatitle','')
    # Vertical color bar
    sargs = dict(
            title=datatitle,
            height=0.67, 
            vertical=True, 
            position_x=0.15, 
            position_y=0.16,
            # Colorbar text properties
            # title_font_size=20,
            # label_font_size=16,
            # shadow=True,
            # n_labels=3,
            # italic=True,
            # fmt="%.1f",
            # font_family="arial",
            )
    plotter.add_mesh(mesh, cmap='jet', scalar_bar_args=sargs)
    plotter.camera_position='xy'
    # plotter.background_color='white'
    # import pdb; pdb.set_trace()    
    filetitle = kwargs.get('filetitle', "Solution at {:.4f}".format(reader.active_time_value))
    prefix,_ = os.path.splitext(filename)
    outputfile = kwargs.get('outputfile', "{:s}_t_{:.4f}".format(prefix,reader.active_time_value))
    outputformat = kwargs.get('outputformat', 'eps')
 
    plotter.save_graphic(outputfile + '.' + outputformat , title=filetitle)

def plot_pvd_2d_color(filename, plottime, dataname):
    import pyvista as pv
    # Details are at https://docs.pyvista.org/api/readers/_autosummary/pyvista.PVDReader.html
    reader = pv.get_reader(filename)
    # Print available time values
    print(reader.time_values)
    if not plottime:
        # Use the first time point 
        reader.set_active_time_point(0)
    else:
        reader.set_active_time_value(plottime)
    mesh = reader.read()[0]
    # Print available point data
    print(mesh.point_data)  
    mesh.plot(cpos='xy',cmap='jet',scalar_bar_args={'title': 'Concentration'})

def plot_pvd_warp(filename, plottime, dataname):
    import pyvista as pv
    # Details are at https://docs.pyvista.org/api/readers/_autosummary/pyvista.PVDReader.html
    reader = pv.get_reader(filename)
    # Print available time values
    print(reader.time_values)
    if not plottime:
        # Use the first time point 
        reader.set_active_time_point(0)
    else:
        reader.set_active_time_value(plottime)
    mesh = reader.read()[0]
    # Print available point data
    print(mesh.point_data)  
    warped = mesh.warp_by_scalar(dataname)
    warped.plot()
