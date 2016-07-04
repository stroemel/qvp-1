# -*- coding: utf-8 -*-
"""

Created on Mon May 26 16:09:26 2014

@author: timo
median averaging for zh, zdr,rho, phi over 360 degrees

"""
# Todo: add wet bulb temperature
# from cosmo data

import warnings

import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
import h5py
import datetime as dt
from scipy import stats
import glob
import os
import wradlib as wrl

warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
-----------------------------------------------------------------
 global data    
-----------------------------------------------------------------
"""
date = '2015-06-22'
location = 'Bonn'
h1, m1 = (15, 00)
h2, m2 = (18, 30)

# path='/home/k.muehlbauer/ownCloud/data'
# path2='/home/k.muehlbauer/ownCloud/prog/snippets/silke/temp'
# path='/automount/radar-archiv/scans/2014/2014-06/2014-06-09'
path = '/mnt/scans/2015/2015-06/2015-06-22'
path2 = '/home/silke/Python/output/Riming'
# Achtung Elevation Angle muß unten auch gesetzt werden
# aendern
file_path = path + '/' + 'n_ppi_280deg/'
# file_path= path  + '/' + '/ppi_8p1deg/'
textfile_path = path2 + '/' + date + '/textfiles/'
plot_path = path2 + '/' + date + '/plots/'

plot_width = 9
plot_height = 7.2

offset_z = 3
offset_phi = 90
offset_zdr = 0.8
special_char = ":"

"""
functions
"""


# -----------------------------------------------------------------
def file_name_list():
    """
    generates a list of filenames for the regarded time interval
    from h1:m1 to h2:m2
    """
    file_names = []
    hour = h1
    minute = m1
    while hour * 60 + minute <= h2 * 60 + m2:
        file_name = generate_file_name(hour, minute)
        file_names.append(file_name)
        hour, minute = next_time(hour, minute)
    return file_names


# ----------------------------------------------------------------
def generate_file_name(hour, minute):
    """
    Generates a data filename out of date and time 
    """
    hour = str(hour)
    if len(hour) == 1:
        hour = "0" + hour
    minute = str(minute)
    if len(minute) == 1:
        minute = "0" + minute
    file_name = date + "--" + hour + special_char + minute + special_char + "00,00.mvol"

    # print "filename: ",file_name
    return file_name


# -----------------------------------------------------------------

def next_time(hour, minute):
    """
    computes the time 5 minutes later
    input: actual time as integers
    return: actual time + 5 minutes as integers
    """
    minute = minute + 5
    if minute >= 60:
        minute = minute - 60  # geändert
        hour = hour + 1
    return hour, minute


# -----------------------------------------------------------------
# -----------------------------------------------------------------
def movingaverage(values, window, mode):
    ''' moving average calculation: 
    Parameters: 
    values: source array  
    window: number of elements for the averaging '''
    weights = np.repeat(1.0, window) / window
    ma = np.convolve(values, weights, mode)
    return ma


def fit_line(x, y):
    ''' slope calculation
    Parameters: x and y array '''
    slope, intercept, r_value, p_value, err = stats.linregress(x, y)
    return slope


def kdp_calc(p, wl=5):
    ''' K_DP calculation for all time steps
    Parameter: p: radar matrix for all time steps '''
    print(p.shape)

    # get x-array over last dimension of source-array (here [2])
    x = np.arange(0, p.shape[2])

    # kdp = np.ones_like(p)*(-10)
    kdp = np.ones_like(p) * (0)

    # iterate over timesteps
    for k in range(np.shape(p)[0]):
        # iterate over azimuths
        for j in range(np.shape(p)[1]):
            # iterate along ray
            for i in range(np.shape(p)[2] - wl):
                # fit
                slope = fit_line(x[i:i + wl - 1], p[k, j, i:i + wl - 1])
                # kdp[k, j, i+wl] = slope/2.
                # Durch 2 wg Def von Kdp, mal 10 da 100m Auflösung und Einheit deg/km
                kdp[k, j, i + 3] = slope / 2. * 10.
    return kdp


# -----------------------------------------------------------------
# -----------------------------------------------------------------
def transform(data, dmin, dmax, dformat):
    """
    transforms the raw data to the dynamic range [dmin,dmax]
    """

    if dformat == 'UV8':
        dform = 255
    else:
        dform = 65535
    # or even better: use numpy arrays, which removes need of for loops
    t = dmin + data * (dmax - dmin) / dform
    return t


# -----------------------------------------------------------------

def read_generic_hdf5(fname):
    """Reads hdf5 files according to their structure

    In contrast to other file readers under wradlib.io, this function will 
    *not* returnreturn a two item tuple with (data, metadata). Instead, this 
    function returns ONE a dictionary that contains all the file contents - 
    both data and metadata. The keys of the output dictionary conform to the 
    Group/Subgroup directory branches of the original file.

    Parameters
    ----------
    fname : string (a hdf5 file path)

    Returns
    -------
    output : a dictionary that contains both data and metadata according to the
              original hdf5 file structure

    """
    f = h5py.File(fname, "r")
    fcontent = {}

    def filldict(x, y):
        # create a new container
        tmp = {}
        # add attributes if present
        if len(y.attrs) > 0:
            tmp['attrs'] = dict(y.attrs)
        # add data if it is a dataset
        if isinstance(y, h5py.Dataset):
            tmp['data'] = np.array(y)
        # only add to the dictionary, if we have something meaningful to add
        if tmp != {}:
            fcontent[x] = tmp

    f.visititems(filldict)

    f.close()
    return fcontent


# -----------------------------------------------------------------

def read_file_data(file_name):
    """
    reads data from hdf5-file
    gelesen werden die Daten zh,phi,rho,zdr mit Zeitstempel.
    Mit der Rückgabe von no_file=0 wird zum Ausdruck gebracht, dass
    die Datei korrekt gelesen werden konnte.
    
    Im Fehlerfall (Rückgabe n0_file=1) werden Dummy-Daten returniert und 
    auf den Bildschirm der Name der nicht gefundenen Datei ausgegeben.    
    """
    dummy = 0
    no_file = 0
    try:
        data = read_generic_hdf5(file_name)

    except:
        print("File " + file_name + " nicht vorhanden.")
        zh = dummy
        phi = dummy
        rho = dummy
        zdr = dummy
        vel = dummy

        no_file = 1
        return dummy, dummy, dummy, dummy, dummy, dummy, no_file, dummy

    zh = transform(data['scan0/moment_10']['data'],
                   data['scan0/moment_10']['attrs']['dyn_range_min'],
                   data['scan0/moment_10']['attrs']['dyn_range_max'],
                   data['scan0/moment_10']['attrs']['format'])

    phi = transform(data['scan0/moment_1']['data'],
                    data['scan0/moment_1']['attrs']['dyn_range_min'],
                    data['scan0/moment_1']['attrs']['dyn_range_max'],
                    data['scan0/moment_1']['attrs']['format'])

    rho = transform(data['scan0/moment_2']['data'],
                    data['scan0/moment_2']['attrs']['dyn_range_min'],
                    data['scan0/moment_2']['attrs']['dyn_range_max'],
                    data['scan0/moment_2']['attrs']['format'])

    zdr = transform(data['scan0/moment_9']['data'],
                    data['scan0/moment_9']['attrs']['dyn_range_min'],
                    data['scan0/moment_9']['attrs']['dyn_range_max'],
                    data['scan0/moment_9']['attrs']['format'])

    vel = transform(data['scan0/moment_5']['data'],
                    data['scan0/moment_5']['attrs']['dyn_range_min'],
                    data['scan0/moment_5']['attrs']['dyn_range_max'],
                    data['scan0/moment_5']['attrs']['format'])

    radar_height = data['where']['attrs']['height']
    s_format = '%SZ'
    if location == 'Juelich':
        s_format = '%S.000Z'

    print(data['scan0/how']['attrs']['timestamp'].decode())
    fstring = '%Y-%m-%dT%H:%M:{0}'.format(s_format)
    print("Time:", dt.datetime.strptime(data['scan0/how']['attrs']['timestamp'].decode(), fstring))

    return zh, phi, rho, zdr, vel, dt.datetime.strptime(data['scan0/how']['attrs']['timestamp'].decode(),
                                                        '%Y-%m-%dT%H:%M:' + s_format), no_file, radar_height


# -----------------------------------------------------------------

def read_file_data2(file_name):
    """
    reads data from hdf5-file
    gelesen werden die Daten zh,phi,rho,zdr mit Zeitstempel.
    Mit der Rückgabe von no_file=0 wird zum Ausdruck gebracht, dass
    die Datei korrekt gelesen werden konnte.
    
    Im Fehlerfall (Rückgabe n0_file=1) werden Dummy-Daten returniert und 
    auf den Bildschirm der Name der nicht gefundenen Datei ausgegeben.    
    """
    dummy = 0
    no_file = 0
    try:
        data = read_generic_hdf5(file_name)

    except:
        print("File " + file_name + " nicht vorhanden.")
        zh = dummy
        phi = dummy
        rho = dummy
        zdr = dummy
        no_file = 1
        return dummy, dummy, dummy, dummy, dummy, no_file, dummy

    zh = transform(data['scan0/moment_10']['data'],
                   data['scan0/moment_10']['attrs']['dyn_range_min'],
                   data['scan0/moment_10']['attrs']['dyn_range_max'],
                   data['scan0/moment_10']['attrs']['format'])

    phi = transform(data['scan0/moment_1']['data'],
                    data['scan0/moment_1']['attrs']['dyn_range_min'],
                    data['scan0/moment_1']['attrs']['dyn_range_max'],
                    data['scan0/moment_1']['attrs']['format'])

    rho = transform(data['scan0/moment_2']['data'],
                    data['scan0/moment_2']['attrs']['dyn_range_min'],
                    data['scan0/moment_2']['attrs']['dyn_range_max'],
                    data['scan0/moment_2']['attrs']['format'])

    zdr = transform(data['scan0/moment_9']['data'],
                    data['scan0/moment_9']['attrs']['dyn_range_min'],
                    data['scan0/moment_9']['attrs']['dyn_range_max'],
                    data['scan0/moment_9']['attrs']['format'])
    radar_height = data['where']['attrs']['height']
    s_format = '%SZ'
    if location == 'Juelich':
        s_format = '%S.000Z'

    return zh, phi, rho, zdr, dt.datetime.strptime( \
        data['scan0/how']['attrs']['timestamp'], '%Y-%m-%dT%H:%M:' + \
                                                 s_format), \
           no_file, radar_height


# -----------------------------------------------------------------

def write_textfile(data, k, filename, bh):
    """
    write plot data into a data table file with time (hh:mm) as 
    x-axis and height (km) as y-axis
    """
    s = []
    y = np.array(["%15.10f" % w for w in data.reshape(data.size)])
    y = y.reshape(data.shape)
    z = y.tolist()
    for n in range(len(z)):
        a = ''.join(str(e) for e in z[n])
        s.append(a)

    datei = open(textfile_path + filename, 'w')
    datei.write(head_lines(k))

    kk = 0
    for line in s:
        new_line = "{:4d}".format(kk) + "{:13.3f}".format(bh[kk]) + line + \
                   '\n'
        datei.write(new_line)
        kk = kk + 1
    datei.close()


# -----------------------------------------------------------------
# -----------------------------------------------------------------
def head_lines(k):
    """
    auxillary function for function write_textfile
    writing the head line
    k: type of data (ZH, PHI_DP, RHO_HV or  ZDR)
    """
    first_lines = 'Radar:' + location + '\n\n''    ' + k + '-DATA    DATE: ' + date + \
                  '\n\n  Bin   Height/km'
    h = h1
    m = m1
    while h * 60 + m <= h2 * 60 + m2:
        hour = str(h)
        if len(hour) == 1:
            hour = "0" + hour
        minute = str(m)
        if len(minute) == 1:
            minute = "0" + minute
        first_lines = first_lines + '     ' + hour + ':' + minute + '     '
        h, m = next_time(h, m)
    first_lines = first_lines + '\n\n'
    return first_lines


# -----------------------------------------------------------------

# -----------------------------------------------------------------
def qvp_Boxpol():
    """
    -----------------------------------------------------------------
     main program
    -----------------------------------------------------------------
    """
    t1 = dt.datetime.now()
    # for noise reduction: only for Bonn
    # wichtig austauschen bei anderen bins
    # aendern
    range_bin_dist = np.arange(50, 35001, 100)
    # range_bin_dist = np.arange(50,100001,100)

    # range_bin_dist also hardcoded here
    # range_bin_dist = np.arange(500,60001,1000)
    # range_bin_dist = np.arange(50,110001,100)

    # aendern
    bins = 350
    # bins=1000

    # bins=60
    # bins=1100
    azi = 360

    file_names = sorted(glob.glob(os.path.join(file_path, '*mvol')))

    print(file_names[0])
    # just to get radar_height
    zh, phi, rho, zdr, vel, dt_src1, no_file, radar_height \
        = read_file_data(file_names[0])

    # brauch' ich die folgenden 2 Zeilen? Analoges für BoXPol? eingelesen wird doch darüber?!
    # odim = wrl.io.read_OPERA_hdf5(file_names[0])
    # print(odim)

    try:
        save = np.load(path2 + '/' + location + '.npz')
    except IOError:
        file_names = sorted(glob.glob(os.path.join(file_path, '*mvol')))

        # geaendert, hier sollen nun nur noch die in der Liste sein, die im gewuenschten Zeitintervall liegen
        # file_names=file_name_list()
        # print(file_names)
        start_time = dt.datetime(2015, 6, 22, 15, 00)
        end_time = dt.datetime(2015, 6,22, 18, 30)

        file_list = []
        for fname in file_names:
            time = dt.datetime.strptime(os.path.splitext(os.path.basename(fname))[0], "%Y-%m-%d--%H:%M:%S,%f")
            if time >= start_time and time <= end_time:
                file_list.append(fname)

        print(file_list)
        file_names = file_list

        #
        n_files = len(file_names)

        result_np_zh = np.zeros((n_files, azi, bins))
        result_np_phi = np.zeros((n_files, azi, bins))
        result_np_rho = np.zeros((n_files, azi, bins))
        result_np_zdr = np.zeros((n_files, azi, bins))
        # -----------------------------------------------------------------
        dt_src = []  # list for time stamps
        # -----------------------------------------------------------------
        # read data files

        for n, fname in enumerate(file_names):
            print("FILENAME:", fname)
            zh, phi, rho, zdr, vel, dt_src1, no_file, radar_height \
                = read_file_data(fname)

            print(dt_src1)
            print("---------> Reading finished")
            """
            #read next file - if there is no file --> (no_file=1):
            """

            if no_file == 1:
                continue
            # add the offset
            zh += offset_z
            phi += offset_phi
            zdr += offset_zdr

            # noise reduction for rho_hv
            # vorher -23
            # vorher -21
            noise_level = -23
            # noise_level = -32

            # aendern
            snr = np.zeros((360, 350))
            # snr = np.zeros((360,1000))
            # snr = np.zeros((360,60))
            # snr = np.zeros((360,1100))

            for i in range(360):
                snr[i, :] = zh[i, :] - 20 * np.log10(range_bin_dist * 0.001) - noise_level - \
                            0.033 * range_bin_dist / 1000
            rho = rho * np.sqrt(1. + 1. / 10. ** (snr * 0.1))

            result_np_zh[n, ...] = zh
            result_np_phi[n, ...] = phi
            result_np_rho[n, ...] = rho
            result_np_zdr[n, ...] = zdr
            dt_src.append(dt_src1)

        # save data to file
        np.savez(path2 + '/' + location, zh=result_np_zh, rho=result_np_rho,
                 zdr=result_np_zdr, phi=result_np_phi, dt_src=dt_src)
        # load again
        save = np.load(path2 + '/' + location + '.npz')

    else:
        # extract data to arrays
        result_data_phi = save['phi']
        result_data_rho = save['rho']
        result_data_zdr = save['zdr']
        result_data_zh = save['zh']
        dt_src = save['dt_src']

        print("Result-Shape:", result_data_phi.shape)

    import copy
    phi = copy.copy(result_data_phi)

    # process phidp
    test = np.zeros_like(phi)
    kdp = np.zeros_like(phi)
    for i, v in enumerate(phi):
        print("Timestep:", i)
        for j, w in enumerate(v):
            ma = movingaverage(w, 11, mode='same')

            # window = [-1, 0, 0, 0, 1]
            # slope = np.convolve(range(len(w)), window, mode='same') / np.convolve(w, window, mode='same')
            # slope, intercept, r_value, p_value, std_err = stats.linregress(range(len(w)),w)
            # slope = np.gradient(w)
            # print(slope.shape)
            # kdp[i, j, :] = slope
            test[i, j, :] = ma
            # test[i, j, result_data_zh[i, j, :] < -5.] = np.nan
            test[i, j, result_data_zh[i, j, :] < -5.] = np.nan
            # print(test[i, j, :])
            first = np.argmax(test[i, j, :] >= 0.)
            last = np.argmax(test[i, j, ::-1] >= 0)
            # print("f:", first, test[i, j, first], test[i, j, first-1])
            # print("l:", last, test[i, j, -last-1], test[i, j, -last])
            if first:
                test[i, j, :first + 1] = test[i, j, first]
            if last:
                test[i, j, -last:] = test[i, j, -last - 1]

    # get kdp from phidp

    # V1: kdp from convolution, maximum speed
    kdp = wrl.dp.kdp_from_phidp_convolution(test, L=3, dr=1.0)

    # V2: fit ala ryzhkov, see function declared above
    #kdp = kdp_calc(test, wl=11)

    print(kdp.shape)

    # median calculation

    result_data_zh = stats.nanmedian(result_data_zh, axis=1).T
    result_data_rho = stats.nanmedian(result_data_rho, axis=1).T
    result_data_zdr = stats.nanmedian(result_data_zdr, axis=1).T

    # mask kdp eventually,
    for i in range(360):
        k1 = kdp[:, i, :]
        k1[result_data_zh.T < -5.] = np.nan
        # print(k1.shape)
        kdp[:, i, :] = k1

    result_data_kdp = stats.nanmedian(kdp, axis=1).T

    # mask phidp eventually,
    for i in range(360):
        k2 = test[:, i, :]
        k2[result_data_zh.T < -5.] = np.nan
        # print(k1.shape)
        test[:, i, :] = k2

    result_data_phi = stats.nanmedian(test, axis=1).T
    # result_data_phi_median = stats.nanmedian(phi, axis=1).T


    print("SHAPE1:", result_data_phi.shape, result_data_zdr.shape)
    # -----------------------------------------------------------------
    # calulate beam_height: array with 350 elements
    # Elevation angle is hardcoded here
    # aendern
    beam_height = (wrl.georef.beam_height_n(np.linspace(0, (bins - 1) * 100, bins), 28.0)
                   + radar_height) / 1000

    # data output in text files
    write_textfile(result_data_zh, 'Z_H', 'zh_output_' + location + '_' + date[0:4] + \
                   '_' + date[5:7] + '_' + date[8:10] + '.txt', beam_height)
    write_textfile(result_data_phi, 'PHI_DP', 'phi_dp_output_' + location + '_' + \
                   date[0:4] + '_' + date[5:7] + '_' + date[8:10] + '.txt', beam_height)
    write_textfile(result_data_rho, 'RHO_HV', 'rho_hv_output_' + location + '_' + \
                   date[0:4] + '_' + date[5:7] + '_' + date[8:10] + '.txt', beam_height)
    write_textfile(result_data_zdr, 'ZDR', 'zdr_output_' + location + '_' + date[0:4] + \
                   '_' + date[5:7] + '_' + date[8:10] + '.txt', beam_height)

    # time stamps
    # print(dt_src)
    # print(dt_src[0], dt_src[-1])
    dt_start = mdates.date2num(dt_src[0])
    dt_stop = mdates.date2num(dt_src[-1])
    # dt_start=start_time
    # dt_stop=end_time
    # contour lines
    # if true plot contours, if false plot pcolormesh
    contour = True  # False
    contourlevels = [0, 5, 10, 20, 30]

    # define the colors
    colors = '#00f8f8', '#01b8fb', '#0000fa', '#00ff00', '#00d000', '#009400', '#ffff00' \
        , '#f0cc00', '#ff9e00', '#ff0000', '#e40000', '#cc0000', '#ff00ff', '#a06cd5'

    # -----------------------------------------------------------------
    # calulate beam_height:
    # Elevation angle
    # geloescht, steht doch ein paar Zeilen weiter oben schon ?!
    # beam_height =( wrl.georef.beam_height_n(np.linspace(0, (bins-1)*1000,bins), 28.0) \
    # + radar_height)/ 1000

    # -----------------------------------------------------------------
    # result_data_... plotting
    fig1 = plt.figure(figsize=(plot_width, plot_height))
    fig2 = plt.figure(figsize=(plot_width, plot_height))
    fig3 = plt.figure(figsize=(plot_width, plot_height))
    fig4 = plt.figure(figsize=(plot_width, plot_height))
    fig5 = plt.figure(figsize=(plot_width, plot_height))
    fig6 = plt.figure(figsize=(plot_width, plot_height))

    ax1 = fig1.add_subplot(111)
    ax1.set_title('$\mathrm{\mathsf{Z_{H}}}$')

    ax2 = fig2.add_subplot(111)
    ax2.set_title('$\mathrm{\mathsf{\phi_{DP}}}$')

    ax3 = fig3.add_subplot(111)
    ax3.set_title(r'$\mathrm{\mathsf{\rho_{hv}}}$')

    ax4 = fig4.add_subplot(111)
    ax4.set_title('$\mathrm{\mathsf{Z_{DR}}}$')

    ax5 = fig5.add_subplot(111)
    ax5.set_title('$\mathrm{\mathsf{K_{DP}}}$')

    # y, x = result_data_zdr.shape
    y = beam_height  # np.arange(x)
    x = mdates.date2num(dt_src)
    X, Y = np.meshgrid(x, y)
    print(X.shape, Y.shape)

    CS1 = ax1.contour(X, Y, result_data_zh, contourlevels, manual='True',
                      origin='lower', colors='k', alpha=0.8,
                      linewidths=1,
                      extent=[dt_start, dt_stop, \
                              # beam_height[0], beam_height[-1]])
                              -0.11, beam_height[-1]])
    ax1.clabel(CS1, fmt='%2.0f', inline=True, fontsize=10)

    CS2 = ax2.contour(X, Y, result_data_zh, contourlevels,
                      origin='lower', colors='k', alpha=1,
                      linewidths=1,
                      extent=[dt_start, dt_stop, \
                              ##beam_height[0], beam_height[-1]])
                              -0.11, beam_height[-1]])

    CS3 = ax3.contour(X, Y, result_data_zh, contourlevels,
                      origin='lower', colors='k',
                      linewidths=1, alpha=1,
                      extent=[dt_start, dt_stop, \
                              # beam_height[0], beam_height[-1]])
                              -0.11, beam_height[-1]])

    CS4 = ax4.contour(X, Y, result_data_zh, contourlevels,
                      origin='lower', colors='k',
                      linewidths=1, alpha=1,
                      extent=[dt_start, dt_stop, \
                              # beam_height[0], beam_height[-1]])
                              -0.11, beam_height[-1]])

    CS5 = ax5.contour(X, Y, result_data_zh, contourlevels,
                      origin='lower', colors='k',
                      linewidths=1, alpha=1,
                      extent=[dt_start, dt_stop, \
                              # beam_height[0], beam_height[-1]])
                              -0.11, beam_height[-1]])

    # define the colors
    colors = '#00f8f8', '#01b8fb', '#0000fa', '#00ff00', '#00d000', '#009400', '#ffff00' \
        , '#f0cc00', '#ff9e00', '#ff0000', '#e40000', '#cc0000', '#ff00ff', '#a06cd5'

    # -----------------------------------------------------------------
    """
    ZDR plot
    """
    levels = [-2, -1, 0, .1, .2, .3, .4, .5, .6, .8, 1.0, 1.2, 1.5, 2.0, 2.5]
    # levels=[-2,-1,-0.6,-0.3,0,.1,.2,.3,.4,.5,.6,.8,1.0,1.2,1.5]
    if contour:
        im4 = ax4.contourf(result_data_zdr, levels=levels, \
                           # im4 = ax4.contourf(np.ma.masked_invalid(result_data_zdr),levels=levels, \
                           colors=colors, origin='lower', axis='equal', extent=[dt_start, dt_stop, \
                                                                                ##beam_height[0], beam_height[-1]])
                                                                                -0.11, beam_height[-1]])
    else:
        cmap = ListedColormap(colors, name='zdr')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im4 = ax4.pcolormesh(X, Y, result_data_zdr, cmap=cmap, norm=norm)

    ax4.set_ylim(0, 13)
    ax4.xaxis_date()
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('\n%H:%M'))
    ax4.xaxis.set_major_locator(mdates.HourLocator())
    ax4.xaxis.set_minor_formatter(mdates.DateFormatter('%M'))
    ax4.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=(15, 30, 45, 60)))
    ax4.grid()
    ax4.set_ylabel('Height (km)')
    ax4.set_xlabel('Time (UTC)')

    cb4 = fig4.colorbar(im4, orientation='vertical', pad=0.018, aspect=35)
    cb4.outline.set_visible(False)
    cbarytks = plt.getp(cb4.ax.axes, 'yticklines')
    plt.setp(cbarytks, visible=False)
    cb4.set_ticks(levels[0:-1])
    cb4.set_ticklabels(["%.2f" % lev for lev in levels[0:-1]])
    cb4.set_label('Differential Reflectivity (dB)')
    # -----------------------------------------------------------------
    """
    ZH plot
    """

    levels = [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 65]

    if contour:
        im1 = ax1.contourf(result_data_zh, levels=levels, \
                           ##im1 = ax1.contourf(np.ma.masked_invalid(result_data_zh),levels=levels,\
                           colors=colors, origin='lower', axis='equal', extent=[dt_start, dt_stop, \
                                                                                ##beam_height[0], beam_height[-1]])
                                                                                -0.11, beam_height[-1]])
    else:
        cmap = ListedColormap(colors, name='zh')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im1 = ax1.pcolormesh(X, Y, result_data_zh, cmap=cmap, norm=norm)

    ax1.set_aspect('auto')
    ax1.set_ylim(0, 13)
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('\n%H:%M'))
    ax1.xaxis.set_major_locator(mdates.HourLocator())
    ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%M'))
    ax1.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=(15, 30, 45, 60)))
    ax1.grid()
    ax1.set_ylabel('Height (km)')
    ax1.set_xlabel('Time (UTC)')

    cb1 = fig1.colorbar(im1, pad=0.018, orientation='vertical', aspect=35)
    cb1.outline.set_visible(False)
    cbarytks = plt.getp(cb1.ax.axes, 'yticklines')
    plt.setp(cbarytks, visible=False)
    cb1.set_ticks(levels[0:-1])
    cb1.set_ticklabels(["%.2f" % lev for lev in levels[0:-1]])
    cb1.set_label('Reflectivity (dBz)')
    # -----------------------------------------------------------------
    """
    PHI_DP plot
    """
    # levels=[0,2,4,6,8,10,12,15,20,25,30,40,50,60,70]
    levels = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 40, 100]
    # levels = np.arange(0,30,3)

    if contour:
        im2 = ax2.contourf(result_data_phi, levels=levels, \
                           # im2 = ax2.contourf(np.ma.masked_invalid(result_data_phi),levels=levels,\
                           colors=colors, origin='lower', axis='equal', extent=[dt_start, dt_stop, \
                                                                                beam_height[0], beam_height[-1]])
        # -0.11, beam_height[-1]])
    else:
        cmap = ListedColormap(colors, name='phi')
        cmap = plt.get_cmap('viridis')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im2 = ax2.pcolormesh(X, Y, result_data_phi, cmap=cmap, norm=norm)
        # im2 = ax2.pcolormesh(result_data_phi, cmap=cmap, norm=norm)

    ax2.set_aspect('auto')
    ax2.set_ylim(0, 13)
    ax2.xaxis_date()
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('\n%H:%M'))
    ax2.xaxis.set_major_locator(mdates.HourLocator())
    ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%M'))
    ax2.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=(15, 30, 45, 60)))
    ax2.grid()
    ax2.set_ylabel('Height (km)')
    ax2.set_xlabel('Time (UTC)')

    cb2 = fig2.colorbar(im2, pad=0.018, orientation='vertical', aspect=35)
    cb2.outline.set_visible(False)
    cbarytks = plt.getp(cb2.ax.axes, 'yticklines')
    plt.setp(cbarytks, visible=False)
    cb2.set_ticks(levels[0:-1])
    cb2.set_ticklabels(["%.2f" % lev for lev in levels[0:-1]])
    cb2.set_label('Differential Phase (deg)')
    # -----------------------------------------------------------------
    """
    RHO_HV plot
    """

    levels = [.7, .8, .85, .9, .92, .94, .95, .96, .97, .98, .985, .99, .995, 1, 1.1]

    if contour:
        im3 = ax3.contourf(result_data_rho, levels=levels, \
                           ##im3 = ax3.contourf(np.ma.masked_invalid(result_data_rho),levels=levels,\
                           colors=colors, origin='lower', axis='equal', extent=[dt_start, dt_stop, \
                                                                                -0.11, beam_height[-1]])
    else:
        cmap = ListedColormap(colors, name='rho')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im3 = ax3.pcolormesh(X, Y, result_data_rho, cmap=cmap, norm=norm)

    ax3.set_aspect('auto')
    ax3.set_ylim(0, 13)
    ax3.xaxis_date()
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('\n%H:%M'))
    ax3.xaxis.set_major_locator(mdates.HourLocator())
    ax3.xaxis.set_minor_formatter(mdates.DateFormatter('%M'))
    ax3.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=(15, 30, 45, 60)))
    ax3.grid()
    ax3.set_ylabel('Height (km)')
    ax3.set_xlabel('Time (UTC)')

    cb3 = fig3.colorbar(im3, orientation='vertical', pad=0.018, aspect=35)
    cb3.outline.set_visible(False)
    cbarytks = plt.getp(cb3.ax.axes, 'yticklines')
    plt.setp(cbarytks, visible=False)
    cb3.set_ticks(levels[0:-1])
    cb3.set_ticklabels(["%.3f" % lev for lev in levels[0:-1]])
    cb3.set_label('Crosscorrelation Coefficient')

    # -----------------------------------------------------------------
    """
    KDP plot
    """
    levels = [-0.5, -0.1, 0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.0, 2., 3., 4.]
    # levels=np.arange(-20,30,1)

    if contour:
        im5 = ax5.contourf(result_data_kdp, levels=levels, \
                           colors=colors, origin='lower', axis='equal', extent=[dt_start, dt_stop, \
                                                                                beam_height[0], beam_height[-1]])
    else:
        cmap = ListedColormap(colors, name='kdp')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im5 = ax5.pcolormesh(X, Y, result_data_kdp, cmap=cmap, norm=norm)

    ax5.set_aspect('auto')
    ax5.set_xlim(dt_src[1], dt_src[-2])
    ax5.set_ylim(0, 13)
    ax5.xaxis_date()
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('\n%H:%M'))
    ax5.xaxis.set_major_locator(mdates.HourLocator())
    ax5.xaxis.set_minor_formatter(mdates.DateFormatter('%M'))
    # ax5.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=(15,30,45,60)))
    ax5.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=(15, 30, 45, 60)))
    ax5.grid()
    ax5.set_ylabel('Height (km)')
    ax5.set_xlabel('Time (UTC)')

    cb5 = fig5.colorbar(im5, pad=0.018, orientation='vertical', aspect=35)
    cb5.outline.set_visible(False)
    cbarytks = plt.getp(cb5.ax.axes, 'yticklines')
    plt.setp(cbarytks, visible=False)
    cb5.set_ticks(levels[0:-1])
    cb5.set_ticklabels(["%.2f" % lev for lev in levels[0:-1]])
    cb5.set_label('Specific differential Phase (deg/km)')

    # _________________________________save plots __________________________________

    fig1.savefig(plot_path + 'zh_' + location + '_' + date + '.pdf', dpi=300, \
                 bbox_inches='tight')
    fig2.savefig(plot_path + 'phi_dp_' + location + '_' + date + '.pdf', dpi=300, \
                 bbox_inches='tight')
    fig3.savefig(plot_path + 'rho_hv_' + location + '_' + date + '.pdf', dpi=300, \
                 bbox_inches='tight')
    fig4.savefig(plot_path + 'zdr_' + location + '_' + date + '.pdf', dpi=300, \
                 bbox_inches='tight')
    fig5.savefig(plot_path + 'kdp_' + location + '_' + date + '.pdf', dpi=300, \
                 bbox_inches='tight')
    # ______________________________________________________________________________

    plt.show()
    t2 = dt.datetime.now()
    print('Elapsed time: %12.7f seconds.' % ((t2 - t1).total_seconds()))


# =======================================================
if __name__ == '__main__':
    qvp_Boxpol()
