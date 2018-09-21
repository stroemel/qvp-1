# -*- coding: utf-8 -*-
"""

Created on Mon May 26 16:09:26 2014

@author: timo
median averaging for zh, zdr,rho, phi over 360 degrees

"""
# Todo: add wet bulb temperature
# from cosmo data

import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
import h5py

import eccodes as codes
import miub_eccodes as mecc


from scipy import stats
import glob
import os
import wradlib as wrl
import matplotlib.colors as col
import matplotlib.ticker as ticker

#import psutil
#process = psutil.Process(os.getpid())

def get_wradlib_cmap():
    startcolor = 'white'
    color1 = '#8ec7ff'
    color2 = 'dodgerblue'
    color3 = 'lime'
    color4 = 'yellow'
    color5 = 'darkorange'
    color6 = 'red'
    endcolor = 'darkmagenta'
    colors = [startcolor, color1, color2, color3, color4, color5, color6, endcolor]
    return col.LinearSegmentedColormap.from_list('wrl1',colors)


"""
-----------------------------------------------------------------
 global data
-----------------------------------------------------------------
"""
# this defines start and end time
# need to be within the same day
start_time = dt.datetime(2014, 10, 7, 0, 00)
end_time = dt.datetime(2014, 10, 7, 3, 30)
#cosmo_end_time = dt.datetime(2014, 8, 26, 21, 00)

date = '{0}-{1:02d}-{2:02d}'.format(start_time.year, start_time.month, start_time.day)
location = 'Bonn'
#cosmo_path = '/automount/cluma04/CNRW/CNRW_4.23/cosmooutput/' \
#             'out_{0}-00/'.format(date)
textfile_path = '/home/silke/Python/projects/climatology/cosmo/'.format(date)
# COSMO prozessing fuer out_2014-08-05-00 bis jetzt
#cosmo_path = '/automount/cluma04/CNRW/CNRW_5.00_grb2/cosmooutput/' \
#             'out_{0}-00/'.format(date)
cosmo_path = '/automount/cluma04/CNRW/CNRW5_output/' \
             'out_{0}-00/'.format(date)
is_cosmo = os.path.exists(cosmo_path)

## read grid from gribfile
# filename = cosmo_path + 'lfff00{0}{1:02d}00'.format(16,0)
# ll_grid = get_grid_from_gribfile(filename)

if is_cosmo:
    # read grid from constants file
    filename = cosmo_path + 'lfff00000000c'
    print(filename)
    rlat = mecc.get_ecc_value_from_file(filename, 'shortName', 'RLAT')
    rlon = mecc.get_ecc_value_from_file(filename, 'shortName', 'RLON')

    ll_grid = np.dstack((rlon, rlat))
    print("rlat, rlon", rlat.shape, rlon.shape)
    print("ll_grid", ll_grid.shape)
    print("so ist rlat", rlat)
    # calculate juxpol grid indices
    boxpol_coords = (7.071663, 50.73052)
    llx = np.searchsorted(ll_grid[0, :, 0], boxpol_coords[0], side='left')
    lly = np.searchsorted(ll_grid[:, 0, 1], boxpol_coords[1], side='left')
    print("Coords Bonn: ({0},{1})".format(llx, lly))

    # read height layers from constants file
    filename = cosmo_path + 'lfff00000000c'
    hhl = mecc.get_ecc_value_from_file(filename,
                                       'shortName',
                                       'HHL')[llx, lly, ...]
    # get km from meters
    hhl = hhl / 1000.

    # reading temperature from associated comso files
    # getting count of comso files
    # beware only available from 00:00 to 21:00
    tcount = int(divmod((end_time - start_time).total_seconds(), 60 * 30)[0] + 2)
    tcount2 = int(divmod((end_time - start_time).total_seconds(), 60 * 15)[0] + 1)
    print(tcount)

    # create timestamps every full 30th minute (00, 30)
    cosmo_dt_arr = [(start_time + dt.timedelta(minutes=30 * i)) for i in range(tcount)]
    cosmo_time_arr = [(start_time + dt.timedelta(minutes=30 * i)).time() for i in range(tcount)]
    print(cosmo_dt_arr)
    # create timestamps every full 15th minute (00, 15)
    cosmo_dt_arr2 = [(start_time + dt.timedelta(minutes=15 * i)) for i in range(tcount2)]
    cosmo_time_arr2 = [(start_time + dt.timedelta(minutes=15 * i)).time() for i in range(tcount2)]

    # create temperature, wind and relative humidity arrays and read from grib files
    temp = np.ones((len(cosmo_time_arr), 50)) * np.nan
    uwind = np.ones((len(cosmo_time_arr), 50)) * np.nan
    vwind = np.ones((len(cosmo_time_arr), 50)) * np.nan

    for it, t in enumerate(cosmo_time_arr):
        filename = cosmo_path + 'lfff00{:%H%M%S}'.format(t)
        print(filename)
        #print("XX:", mecc.get_ecc_value_from_file(filename, 'shortName', 'relhum').shape)
        try:
            temp[it, ...] = mecc.get_ecc_value_from_file(filename, 'shortName', 't')[llx, lly, ...]
            #vmax[it, ...] = mecc.get_ecc_value_from_file(filename, 'shortName', 'vmax_10m')[llx, lly]
            #relh[it, ...] = mecc.get_ecc_value_from_file(filename, 'shortName', 'RELHUM_2M')[llx, lly]
            uwind[it, ...] = mecc.get_ecc_value_from_file(filename, 'shortName', 'u')[llx, lly, ...]
            vwind[it, ...] = mecc.get_ecc_value_from_file(filename, 'shortName', 'v')[llx, lly, ...]
        except IOError:
            pass
    vmax = np.ones((len(cosmo_time_arr2))) * np.nan
    relh = np.ones((len(cosmo_time_arr2))) * np.nan
    for it, t in enumerate(cosmo_time_arr2):
        filename = cosmo_path + 'lfff00{:%H%M%S}'.format(t)
        print(filename)
        # print("XX:", mecc.get_ecc_value_from_file(filename, 'shortName', 'relhum').shape)
        try:
            # vmax[it, ...] = mecc.get_ecc_value_from_file(filename, 'shortName', 'vmax_10m')[llx, lly]
            relh[it, ...] = mecc.get_ecc_value_from_file(filename, 'shortName', 'RELHUM_2M')[llx, lly]
        except IOError:
            pass
    # we need degree celsius, not kelvins
    temp = temp - 273.15
    print("XX:", relh)
    wind= np.sqrt(uwind**2.+vwind**2)
    # text output temperature
    fn = '{0}_output_{1}_{2}.txt'.format('temp', location, date)

    header = "Temperature: {0}\tDATE: {1}\n".format(
        location, date)
    y_hhl = np.diff(hhl) / 2 + hhl[:-1]

    print(y_hhl.shape, temp)

    index, tindex = temp.T.shape
    cosmo_time_arr = [(start_time + dt.timedelta(minutes=30 * i)).time() for
                      i in range(tindex)]


# -----------------------------------------------------------------
# result_data_... plotting



# contour lines
# if true plot contours, if false plot pcolormesh
contour = True

# define the colors
colors = '#00f8f8', '#01b8fb', '#0000fa', '#00ff00', '#00d000', '#009400', \
         '#ffff00', '#f0cc00', '#ff9e00', '#ff0000', '#e40000', '#cc0000', \
         '#ff00ff', '#a06cd5'



if is_cosmo:
    # x-y-grid for grib data
    # we use hhl, so we have to calculate the mid of the layers.
    y_hhl = np.diff(hhl) / 2 + hhl[:-1]
    print(y_hhl.shape, hhl.shape)
    x_temp = mdates.date2num(cosmo_dt_arr)
    x_temp2 = mdates.date2num(cosmo_dt_arr2)
    #x_temp = cosmo_dt_arr

    X1, Y1 = np.meshgrid(x_temp, y_hhl)
    #Try plotting with gridspec to achieve that both plots fit nicely concerning horizontal dimension even though one has
    #a colorbar and the other not.
    fig = plt.figure(figsize=(20, 10))
    # erzeuge GridSpec mit 2 Zeilen und 2 Spalten
    # mache die zweite Spalte sehr klein (z.B. 0.02)
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.02])
    # erste axis (links oben)
    ax1 = plt.subplot(gs[0])
    # zweite axis (links unten)
    ax2 = plt.subplot(gs[2])
    # dritte axis (rechts oben, colorbar)
    ax3 = plt.subplot(gs[1])

    #fig, ax = plt.subplots(2, 1, figsize=(20, 10))
    #ax = fig.add_subplot(111)
    ax1.set_title("Wind, Temperature")
    ax2.set_title("Rel. humidity")
    ax1.set_ylim([0, 8])

    cmin = 0
    cmax = 36
    cdiv = 3
    ticks = [i for i in range(cmin, cmax + cdiv, cdiv)]
    bounds = np.concatenate((np.array([cmin - cdiv]), ticks, np.array([cmax + cdiv])), axis=0)
    cmap = get_wradlib_cmap()
    norm = col.BoundaryNorm(bounds, cmap.N)
    b=ax1.pcolormesh(X1,Y1,wind.T,norm=norm, cmap=cmap)
    plt.colorbar(b,cax=ax3)
    cs = ax1.contour(X1, Y1, temp.T, [-15, -10, -5, 0], manual='True',
                origin='lower', colors='k', alpha=0.8,
                linewidths=2)
    ax1.clabel(cs, fmt='%2.0f', inline=True, fontsize=10)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('\n%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator())
    ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%M'))
    ax1.xaxis.set_minor_locator(
        mdates.MinuteLocator(byminute=(15, 30, 45, 60)))
    ax1.set_xlim([x_temp2[0], x_temp2[-1]])
    ax2.plot(x_temp2, relh.T)
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('\n%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator())
    ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%M'))
    ax2.xaxis.set_minor_locator(
        mdates.MinuteLocator(byminute=(15, 30, 45, 60)))
    ax2.set_xlim([x_temp2[0], x_temp2[-1]])
    #plt.tight_layout()

    fig.savefig('/home/silke/Python/projects/climatology/cosmo/' + date + '.png')
    plt.show()
