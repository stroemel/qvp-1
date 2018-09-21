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
import scipy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm
from matplotlib.colors import ListedColormap
import h5py

import eccodes as codes
import miub_eccodes as mecc


from scipy import stats
from scipy import ndimage
import glob
import os
import wradlib as wrl
import math
import pandas as pd

import psutil
process = psutil.Process(os.getpid())

"""
-----------------------------------------------------------------
 global data    
-----------------------------------------------------------------
"""
# this defines start and end time
# need to be within the same day

start_time = dt.datetime(2014, 10, 7, 00, 00)
end_time = dt.datetime(2014, 10, 7, 3,30)

#start_time = dt.datetime(2015, 3, 29, 9, 00)
#end_time = dt.datetime(2015, 3, 29, 13, 20)

#start_time = dt.datetime(2013, 4, 12, 0, 00)
#end_time = dt.datetime(2013, 4, 12, 4, 10)

#start_time = dt.datetime(2014, 11, 16, 0, 00)
#end_time = dt.datetime(2014, 11, 16, 9, 20)


#cosmo_end_time = dt.datetime(2014, 8, 26, 21, 00)

date = '{0}-{1:02d}-{2:02d}'.format(start_time.year, start_time.month, start_time.day)
location = 'Bonn'
#radar_path='/automount/radar/scans/{0}/{0}-{1:02}/{2}'.format(start_time.year, start_time.month, date)
#radar_path='/home/silke/Python/testdata/QVP'
radar_path='/automount/radar-archiv/scans/{0}/{0}-{1:02}/{2}'.format(start_time.year, start_time.month, date)
output_path = '../../output/Riming'
# choose scan
#file_path = radar_path + '/' + 'n_ppi_110deg/'
#nur fuer Check von NBF mit 14Grad elevation, sonst 18
file_path = radar_path + '/' + 'n_ppi_180deg/'
#file_path='/home/silke/Python/testdata/QVP/'
#file_path = radar_path + '/' + 'n_ppi_140deg/'
#file_path = radar_path + '/' + 'n_ppi_280deg/'
textfile_path = output_path + '/{0}/textfiles2/'.format(date)
plot_path = output_path + '/{0}/plots2/'.format(date)

print(radar_path)
print(output_path)
print(textfile_path)
print(plot_path)
#exit(9)
# create paths accordingly
if not os.path.isdir(output_path):
    os.makedirs(output_path)
if not os.path.isdir(textfile_path):
    os.makedirs(textfile_path)
if not os.path.isdir(plot_path):
    os.makedirs(plot_path)

plot_width = 9
plot_height = 7.2

#fuer 12 Apr 2013
offset_z = 3
offset_phi = 90
offset_zdr = 1.2

#fuer 16 Nov 2014
#offset_z = 3
#offset_phi = 85
#offset_zdr = 0.5

#fuer 7 Oct 2014
offset_z = 3
offset_phi = 85
offset_zdr = 0.5

special_char = ":"

"""
functions
"""

# transforms rotated_ll to latlon and vica versa
def rotated_grid_transform(grid_in, option, SP_coor):

    lon = grid_in[...,0]
    lat = grid_in[...,1]

    lon = np.deg2rad(lon) # Convert degrees to radians
    lat = np.deg2rad(lat)

    SP_lon = SP_coor[0]
    SP_lat = SP_coor[1]

    theta = 90 + SP_lat # Rotation around y-axis
    phi = SP_lon # Rotation around z-axis

    phi = np.deg2rad(phi) # Convert degrees to radians
    theta = np.deg2rad(theta)

    x = np.cos(lon) * np.cos(lat) # Convert from spherical to cartesian coordinates
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)

    if not option: # Regular -> Rotated

        x_new = np.cos(theta) * np.cos(phi) * x + np.cos(theta) * np.sin(phi) * y + np.sin(theta) * z
        y_new = -np.sin(phi) * x + np.cos(phi) * y
        z_new = -np.sin(theta) * np.cos(phi) * x - np.sin(theta) * np.sin(phi) * y + np.cos(theta) * z

    else:

        phi = -phi
        theta = -theta

        x_new = np.cos(theta) * np.cos(phi) * x + np.sin(phi) * y + np.sin(theta) * np.cos(phi) * z
        y_new = -np.cos(theta) * np.sin(phi) * x + np.cos(phi) * y - np.sin(theta) * np.sin(phi) * z
        z_new = -np.sin(theta) * x + np.cos(theta) * z

    lon_new = np.arctan2(y_new, x_new) # Convert cartesian back to spherical coordinates
    lat_new = np.arcsin(z_new)

    # +90 added for proper presentation in europe
    lon_new = np.rad2deg(lon_new) + 90  # Convert radians back to degrees
    lat_new = np.rad2deg(lat_new)

    return np.dstack((lon_new, lat_new))

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

class boxpol(object):
    """
    reads data from hdf5-file
    gelesen werden die Daten zh,phi,rho,zdr mit Zeitstempel.
    Mit der Rückgabe von no_file=0 wird zum Ausdruck gebracht, dass
    die Datei korrekt gelesen werden konnte.

    Im Fehlerfall (Rückgabe n0_file=1) werden Dummy-Daten returniert und
    auf den Bildschirm der Name der nicht gefundenen Datei ausgegeben.
    """
    def __init__(self, filename, **kwargs):
        data = read_generic_hdf5(filename)
        #print(data)
        #exit(9)
        if data is not None:
            self._zh = transform(data['scan0/moment_10']['data'],
                           data['scan0/moment_10']['attrs']['dyn_range_min'],
                           data['scan0/moment_10']['attrs']['dyn_range_max'],
                           data['scan0/moment_10']['attrs']['format'])

            self._phi = transform(data['scan0/moment_1']['data'],
                            data['scan0/moment_1']['attrs']['dyn_range_min'],
                            data['scan0/moment_1']['attrs']['dyn_range_max'],
                            data['scan0/moment_1']['attrs']['format'])

            self._rho = transform(data['scan0/moment_2']['data'],
                            data['scan0/moment_2']['attrs']['dyn_range_min'],
                            data['scan0/moment_2']['attrs']['dyn_range_max'],
                            data['scan0/moment_2']['attrs']['format'])

            self._zdr = transform(data['scan0/moment_9']['data'],
                            data['scan0/moment_9']['attrs']['dyn_range_min'],
                            data['scan0/moment_9']['attrs']['dyn_range_max'],
                            data['scan0/moment_9']['attrs']['format'])

            self._vel = transform(data['scan0/moment_5']['data'],
                    data['scan0/moment_5']['attrs']['dyn_range_min'],
                    data['scan0/moment_5']['attrs']['dyn_range_max'],
                    data['scan0/moment_5']['attrs']['format'])

            self._radar_height = data['where']['attrs']['height']

            self._range_samples = data['scan0/how']['attrs']['range_samples']
            self._range_step = data['scan0/how']['attrs']['range_step']
            self._bin_count = data['scan0/how']['attrs']['bin_count']
            self._elevation = data['scan0/how']['attrs']['elevation']

            try:
                self._date = dt.datetime.strptime(data['scan0/how']['attrs']['timestamp'].decode(), '%Y-%m-%dT%H:%M:%SZ')
            except:
                self._date = dt.datetime.strptime(data['scan0/how']['attrs']['timestamp'].decode(), '%Y-%m-%dT%H:%M:%S.000Z')

    @property
    def zh(self):
        """ Returns DataSource
        """
        return self._zh

    @zh.setter
    def zh(self, value):
        self._zh = value

    @property
    def phi(self):
        """ Returns DataSource
        """
        return self._phi

    @phi.setter
    def phi(self, value):
        self._phi = value

    @property
    def rho(self):
        """ Returns DataSource
        """
        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = value

    @property
    def zdr(self):
        """ Returns DataSource
        """
        return self._zdr

    @zdr.setter
    def zdr(self, value):
        self._zdr = value

    @property
    def date(self):
        """ Returns DataSource
        """
        return self._date

    @property
    def radar_height(self):
        """ Returns DataSource
        """
        return self._radar_height

    @property
    def range_step(self):
        """ Returns DataSource
        """
        return self._range_step

    @property
    def bin_count(self):
        """ Returns DataSource
        """
        return self._bin_count

    @property
    def range_samples(self):
        """ Returns DataSource
        """
        return self._range_samples

    @property
    def elevation(self):
        """ Returns DataSource
        """
        return self._elevation


def fig_ax(title, w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    return fig, ax


def melting_layer_qvp(qvp_zh, qvp_zdr, qvp_rhohv, elevation, dr, nbins, height_limit=""):
    """
    A melting layer detection method based on QVP data of ZH, ZDR, and RHOHV. Returns the location of the
    top and bottom of the melting layer.

    Inputs are....
    -QVP data of ZH, ZDR, and RHOhv (qvp_zh,qvp_zdr,qvp_rhohv).where x is height, and y is
    -The elevation angle of the qvps (elevation).
    -The range step (dr)
    -The number of bins for range(bins)
    -Height_limit is optional limit. It will cut the data to certain height. This can improve the functioning because sometimes the top of the clouds produce problems.

    Created on Tuesday April 11 2017

    bhickman@uni-bonn.de


    """
    re = 6374000.
    ke = 4. / 3.
    range_bin_dist = 100.
    radar_height = 99.5
    print(type(nbins))
    r = np.linspace(0, (nbins - 1) * 100, nbins)
    # beam_height=(np.sqrt( r**2 + (ke*re)**2 + 2*r*ke*re*np.sin(np.radians(elevation)) )- ke*re)/1000
    #beam_height = (wrl.georef.beam_height_n(r, round(elevation, 1))
    #               + radar_height) / 1000
    beam_height = wrl.georef.bin_altitude(r, round(elevation, 1), radar_height, re) / 1000

    ############################################################
    # Step 1Normalize QVP data

    # initiating empty arraysfor normalized data
    z0 = np.zeros((qvp_zh.shape[0], qvp_zh.shape[1]))
    rho0 = np.zeros((qvp_zh.shape[0], qvp_zh.shape[1]))
    zdr0 = np.zeros((qvp_zh.shape[0], qvp_zh.shape[1]))

    for ii in range(z0.shape[1]):
        # normalize Z
        z = np.copy(qvp_zh[:, ii])
        zmask = np.where((z < -10) | (z > 60))
        z[zmask] = np.nan
        z0[:, ii] = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z))

        # normalize rho
        rho = np.copy(qvp_rhohv[:, ii])
        rhomask = np.where((rho < 0.65) | (rho > 1))  # rho thresholds
        rho[rhomask] = np.nan
        rho0[:, ii] = (rho - np.nanmin(rho)) / (np.nanmax(rho) - np.nanmin(rho))

        # normalize ZDR
        zd = np.copy(qvp_zdr[:, ii])
        zdmask = np.where((zd < -1) | (zd > 4))
        zd[zdmask] = np.nan
        zdr0[:, ii] = (zd - np.nanmin(zd)) / (np.nanmax(zd) - np.nanmin(zd))

    ############################################################
    # removing profiles with too few data (>93% of array is nan)
    per = np.zeros(z0.shape[1])
    for ii in range(z0.shape[1]):
        zt = z0[:, ii]
        tnan = np.count_nonzero(np.isnan(zt))
        tot = np.float(z0.shape[0])
        per[ii] = tnan / tot * 100
    for ii in range(z0.shape[1]):
        if per[ii] >= 93.:
            z0[:, ii] = np.nan
            zdr0[:, ii] = np.nan
            rho0[:, ii] = np.nan
        else:
            z0[:, ii] = z0[:, ii]
            zdr0[:, ii] = zdr0[:, ii]
            rho0[:, ii] = rho0[:, ii]

            ############################################################
    #### Step 2combining three normalized variables into single varible (IMcomb) ####
    # IMcomb =(zdr0*(1-rho0)*z0)
    ## A try for cases where identified bottom is above top of ML
    IMcomb = (zdr0 * zdr0 * (1 - rho0) * z0)

    ############################################################
    #### STEP 3 SOBEL Filter ####
    dy = ndimage.sobel(IMcomb, 0)

    ############################################################
    #### STEP 4 Threshold ####
    ml_mask = np.where(np.abs(dy) < .02)
    dy[ml_mask] = 0.

    ############################################################
    # Step 4b Height mask
    if not height_limit:
        height_limit = ""
    else:
        height_limit = np.asfarray(height_limit)
        height_mask = np.where(beam_height > height_limit)
        dy[height_mask] = np.nan

    ############################################################
    #### Step 5 ML Height min and max ####
    mlh_ind = np.zeros(IMcomb.shape[1])
    ml_top = np.zeros(dy.shape[1])
    ml_bottom = np.zeros(dy.shape[1])
    mlh_top = np.zeros(dy.shape[1])
    mlh_bottom = np.zeros(dy.shape[1])
    for ii in range(dy.shape[1]):
        d = dy[:, ii]
        # print np.nansum(d), ii
        if np.nansum(d) == math.isnan(np.nansum(d)):
            ml_bottom[ii] = -999
        else:
            ml_bottom[ii] = np.nanargmax(d)
        if np.nansum(d) == math.isnan(np.nansum(d)):
            ml_top[ii] = 999
        else:
            ml_top[ii] = np.nanargmin(d)

    for ii in range(dy.shape[1]):
        if ml_bottom[ii] == -999:
            mlh_bottom[ii] = -999
        else:
            # print("ml_bottom[ii]",ml_bottom[ii])
            mlh_bottom[ii] = beam_height[int(ml_bottom[ii])]

    mlh_bottom[mlh_bottom == -999] = np.nan
    for ii in range(dy.shape[1]):
        if ml_top[ii] == 999:
            mlh_top[ii] = 999
        else:
            mlh_top[ii] = beam_height[int(ml_top[ii])]
    mlh_top[mlh_top == 999] = np.nan

    # Remove MLH_top which are below the MLH_bottom
    for ii in range(len(mlh_bottom)):
        if mlh_top[ii] <= mlh_bottom[ii]:
            mlh_top[ii] = np.nan

    ############################################################
    #### Step 6 Median ML ######
    MED_mlh_bot = pd.rolling_median(mlh_bottom, min_periods=1, center=False, window=36)
    MED_mlh_top = pd.rolling_median(mlh_top, min_periods=1, center=False, window=36)

    if qvp_zh.ndim > 1:
        bh = np.array([beam_height, ] * qvp_zh.shape[1]).transpose()
    else:
        bh = beam_height

    ############################################################
    # Step 7: Step 5 is run again, but this time after discarding the gradient image above ...
    # (1 + fML,height) · MED_mlh_top and below (1 − fML,height ) · MED_mlh_bot , assuming the ML is a relatively
    # flat structure. This helps to remove the possible contamination by ground echoes or small embedded cells
    # of intense rainfall. The chosen value for fML,height is 0.3
    mlh_ind = np.zeros(IMcomb.shape[1])

    ml_top = np.zeros(IMcomb.shape[1])
    ml_bottom = np.zeros(IMcomb.shape[1])

    fMLH = 0.3
    IMabove = (1 + fMLH) * MED_mlh_top
    IMbelow = (1 - fMLH) * MED_mlh_bot

    h_ind = np.where((bh > IMabove) | (bh < IMbelow))
    ml_new = np.copy(dy)
    ml_new[h_ind] = np.nan
    mlh_top = np.zeros(ml_new.shape[1])
    mlh_bottom = np.zeros(ml_new.shape[1])
    for ii in range(dy.shape[1]):
        d = ml_new[:, ii]
        if np.nansum(d) == math.isnan(np.nansum(d)):
            ml_bottom[ii] = -999
        else:
            ml_bottom[ii] = np.nanargmax(d)
        if np.nansum(d) == math.isnan(np.nansum(d)):
            ml_top[ii] = 999
        else:
            ml_top[ii] = np.nanargmin(d)

    for ii in range(dy.shape[1]):
        if ml_bottom[ii] == -999:
            mlh_bottom[ii] = -999
        else:
            mlh_bottom[ii] = beam_height[int(ml_bottom[ii])]
    mlh_bottom[mlh_bottom == -999] = np.nan

    for ii in range(ml_new.shape[1]):
        if ml_top[ii] == 999:
            mlh_top[ii] = 999
        else:
            mlh_top[ii] = beam_height[int(ml_top[ii])]
    mlh_top[mlh_top == 999] = np.nan

    # Remove MLH_top which are below the MLH_bottom
    for ii in range(len(mlh_bottom)):
        if mlh_top[ii] <= mlh_bottom[ii]:
            # A try for very low MLs
            # mlh_top[ii]=np.nan
            # mlh_bottom[ii]=np.nan
            mlh_bottom[ii] = beam_height[int(1)]
            ml_bottom[ii] = 1

    # print 'MLH bottom', np.nanmedian(mlh_bottom), 'MLH top', np.nanmedian(mlh_top)
    # print 'MLH bottom',np.nanmedian(mlh_bottom2),'MLH top', np.nanmedian(mlh_top2)

    # MED_mlh_bot=pd.rolling_median(mlh_bottom,min_periods=1,center=False,window=12)
    # MED_mlh_top=pd.rolling_median(mlh_top,min_periods=1,center=False,window=12)


    return mlh_top, mlh_bottom, ml_top, ml_bottom, beam_height

def add_contour(ax, X, Y, data, levels, **kwargs):
    cs = ax.contour(X, Y, data, levels, **kwargs)
    ax.clabel(cs, fmt='%2.0f', inline=True, fontsize=10)

def add_plot(mom, cfg):
    ax = mom['ax']
    levels = mom['levels']
    if cfg['contour']:
        im = ax.contourf(cfg['X'], cfg['Y'], mom['data'], levels=levels,
                         colors=cfg['colors'], origin='lower', axis='equal',
                         extent=[cfg['dt_start'], cfg['dt_stop'],
                                 -0.11, cfg['beam_height'][-1]])
    else:
        cmap = ListedColormap(cfg['colors'])  # , name='')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im = ax.pcolormesh(cfg['X'], cfg['Y'], mom['data'], cmap=cmap, norm=norm)

    ax.set_ylim(0, 8)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator())
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%M'))
    ax.xaxis.set_minor_locator(
        mdates.MinuteLocator(byminute=(15, 30, 45, 60)))
    ax.grid()
    ax.set_ylabel('Height (km)')
    ax.set_xlabel('Time (UTC)')

    cb = mom['fig'].colorbar(im, orientation='vertical', pad=0.018, aspect=35)
    cb.outline.set_visible(False)
    cbarytks = plt.getp(cb.ax.axes, 'yticklines')
    plt.setp(cbarytks, visible=False)
    cb.set_ticks(levels[0:-1])
    #cb.set_ticklabels(["%.2f" % lev for lev in levels[0:-1]])
    if mom['name'] == 'rho':
        cb.set_ticklabels(["%.3f" % lev for lev in levels[0:-1]])
    else:
        cb.set_ticklabels(["%.2f" % lev for lev in levels[0:-1]])

    cb.set_label(mom['cb_label'])


def get_grid_from_gribfile(filename, rotated=False):

    f = open(filename)
    # get grib message count and create gid_list, close filehandle
    msg_count = codes.codes_count_in_file(f)
    gid_list = [codes.codes_grib_new_from_file(f) for i in range(msg_count)]
    f.close()

    print("Working on grib-file: {0}".format(filename))
    print("Message Count: {0}".format(msg_count))

    # read grib grid details from given gid
    gid = gid_list[0]

    return get_grid_from_gid(gid, rotated=rotated)


def get_grid_from_gid(gid, rotated=False):

    Ni = codes.codes_get(gid, 'Ni')
    Nj = codes.codes_get(gid, 'Nj')
    lat_start = codes.codes_get(gid, 'latitudeOfFirstGridPointInDegrees')
    lon_start = codes.codes_get(gid, 'longitudeOfFirstGridPointInDegrees')
    lat_stop = codes.codes_get(gid, 'latitudeOfLastGridPointInDegrees')
    lon_stop = codes.codes_get(gid, 'longitudeOfLastGridPointInDegrees')

    print("LL: ({0},{1})".format(lon_start, lat_start))
    print("UR: ({0},{1})".format(lon_stop, lat_stop))

    lat_sp = codes.codes_get(gid, 'latitudeOfSouthernPole')
    lon_sp = codes.codes_get(gid, 'longitudeOfSouthernPole')
    ang_rot = codes.codes_get(gid, 'angleOfRotation')

    print("SP: ({0},{1}) - Ang:{2}".format(lon_sp, lat_sp, ang_rot))

    # create grid arrays from grid details
    # iarr, jarr one-dimensional data
    iarr = np.linspace(lon_start, lon_stop, num=Ni, endpoint=False)
    jarr = np.linspace(lat_start, lat_stop, num=Nj, endpoint=False)
    # converted by meshgrid to 2d-arrays
    i2d, j2d = np.meshgrid(iarr, jarr)

    grid_rot = np.dstack((i2d, j2d))

    if not rotated:
        return rotated_grid_transform(grid_rot, 1, [lon_sp, lat_sp])
    else:
        return grid_rot

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

    print(file_path)
    file_names = sorted(glob.glob(os.path.join(file_path, '*mvol')))


    print(file_names[0])

    # get some specifics of the radar data
    ds0 = boxpol(file_names[0])
    bin_range = ds0.range_samples * ds0.range_step
    bin_count = ds0.bin_count
    range_bin_dist = np.arange(bin_range/2, bin_range*bin_count+1, bin_range)
    elevation = ds0.elevation
    # get bins, azi from fist file
    (azi, bins) = ds0.zh.shape

    # try to load existing data
    try:
        save = np.load(output_path + '/' + location + '_' + str(round(elevation, 1)) + '_' + date + '.npz')

        result_data_phi = save['phi']
        result_data_rho = save['rho']
        result_data_zdr = save['zdr']
        result_data_zh = save['zh']
        radar_height = save['radar_height']
        dt_src = save['dt_src']
    # or create data from scratch
    except IOError:
        file_names = sorted(glob.glob(os.path.join(file_path, '*mvol')))
        print(file_names)

        file_list = []
        for fname in file_names:
            time = dt.datetime.strptime(os.path.splitext(os.path.basename(fname))[0], "%Y-%m-%d--%H:%M:%S,%f")
            if time >= start_time and time <= end_time:
                file_list.append(fname)

        print(file_list)
        file_names = file_list
        n_files = len(file_names)

        # define result arrays
        result_data_zh = np.zeros((n_files, azi, bins))
        result_data_phi = np.zeros((n_files, azi, bins))
        result_data_rho = np.zeros((n_files, azi, bins))
        result_data_zdr = np.zeros((n_files, azi, bins))
        # -----------------------------------------------------------------
        dt_src = []  # list for time stamps
        # -----------------------------------------------------------------

        # iterate over files and read data files

        for n, fname in enumerate(file_names):
            print("FILENAME:", fname)

            dsl = boxpol(fname)

            print(dsl.date)
            print("ZH:", dsl.zh.shape)
            print("RES:", result_data_zh.shape)
            print("---------> Reading finished")

            # add the offset
            dsl.zh += offset_z
            dsl.phi += offset_phi
            dsl.zdr += offset_zdr

            # noise reduction for rho_hv

            noise_level = -22
            # fuer 16 Nov 2014
            # noise_level = -22
            # vorher -23, -22 probiert zu hohen rho
            #genommen fuer 2014-08-26
            # fuer Florian angegeben noise_level = -23
            # noise_level = -32

            # aendern
            snr = np.zeros((azi, bins))
            # snr = np.zeros((360,1000))
            # snr = np.zeros((360,60))
            # snr = np.zeros((360,1100))

            for i in range(azi):
                snr[i, :] = dsl.zh[i, :] - 20 * np.log10(range_bin_dist * 0.001) - noise_level - \
                            0.033 * range_bin_dist / 1000
            dsl.rho = dsl.rho * np.sqrt(1. + 1. / 10. ** (snr * 0.1))

            result_data_zh[n, ...] = dsl.zh
            result_data_phi[n, ...] = dsl.phi
            result_data_rho[n, ...] = dsl.rho
            result_data_zdr[n, ...] = dsl.zdr
            radar_height = dsl.radar_height
            dt_src.append(dsl.date)

        # save data to file
        np.savez(output_path + '/' + location + '_' + str(round(elevation, 1)) + '_' + date, zh=result_data_zh, rho=result_data_rho,
                 zdr=result_data_zdr, phi=result_data_phi, dt_src=dt_src, radar_height=radar_height)

    print("Result-Shape:", result_data_phi.shape)

    # try top read phi and kdp from file
    try:
        save = np.load(output_path + '/' + location + '_' + str(round(elevation, 1)) + '_' + date + '_phidp_kdp.npz')
        phi = save['phi']
        kdp = save['kdp']
        test = save['test']
    # or create from scratch
    except:
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

                # Masken vereinheitlichen ?
                test[i, j, result_data_rho[i, j, :] < 0.7] = np.nan
                test[i, j, result_data_zh[i, j, :] < -10.] = np.nan
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
        #vorher mit L=11, ersetzt
        #kdp = wrl.dp.kdp_from_phidp_convolution(test, L=21, dr=0.1)
        #durch
        kdp=wrl.dp.kdp_from_phidp(test, winlen=21, dr=0.1)

        # V2: fit ala ryzhkov, see function declared above
        #kdp = kdp_calc(test, wl=11)

        print(kdp.shape)

        # save data to file
        np.savez(output_path + '/' + location + '_' + str(round(elevation, 1)) + '_' + date + '_phidp_kdp', phi=phi, kdp=kdp, test=test)
    # Versuche vorher zu maskieren
    #mask_ind = np.where(result_data_rho <= 0.85)
    #result_data_zh[mask_ind] = np.nan
    #result_data_rho[mask_ind] = np.nan
    #result_data_zdr[mask_ind] = np.nan

    # Maskieren neu
    # mask rho
    rho_mask = np.ma.masked_less(result_data_rho, 0.7)
    result_data_zh = np.ma.array(result_data_zh, mask=rho_mask.mask)
    result_data_rho = np.ma.array(result_data_rho, mask=rho_mask.mask)
    result_data_zdr = np.ma.array(result_data_zdr, mask=rho_mask.mask)


    # median calculation
    result_data_zh = np.nanmedian(result_data_zh, axis=1).T
    result_data_rho = np.nanmedian(result_data_rho, axis=1).T
    result_data_zdr = np.nanmedian(result_data_zdr, axis=1).T

    # mask kdp eventually,
    for i in range(360):
        k1 = kdp[:, i, :]
        #Masken vereinheitlichen ?
        rho_isnan = np.isnan(result_data_rho.T)
        k1 = np.ma.array(k1, mask=rho_isnan)
        k1[result_data_zh.T < -10.] = np.nan
        # print(k1.shape)
        kdp[:, i, :] = k1

    result_data_kdp = np.nanmedian(kdp, axis=1).T


    # mask phidp eventually,
    for i in range(360):
        k2 = test[:, i, :]
        # Masken vereinheitlichen ?
        k2 = np.ma.array(k2, mask=rho_isnan)
        k2[result_data_zh.T < -10.] = np.nan
        # print(k1.shape)
        test[:, i, :] = k2

    result_data_phi = np.nanmedian(test, axis=1).T
    # result_data_phi_median = stats.nanmedian(phi, axis=1).T

    print("SHAPE1:", result_data_phi.shape, result_data_zdr.shape)
    # -----------------------------------------------------------------
    # calulate beam_height: array with 350 elements
    # Elevation angle is hardcoded here
    # aendern
    #beam_height = (wrl.georef.beam_height_n(np.linspace(0, (bins - 1) * 100, bins), 28.0)
    #               + radar_height) / 1000

    #beam_height = (wrl.georef.beam_height_n(range_bin_dist, round(elevation,1))
    #               + radar_height) / 1000
    re = 6374000.
    beam_height = wrl.georef.bin_altitude(range_bin_dist, round(elevation, 1), radar_height, re) / 1000
    # Calculate new kdp, based on modified phidp which does not include delta in the ML
    ############################################################
    #modified phidp
    print("SHAPES:", result_data_phi.shape, result_data_kdp.shape, result_data_zh.shape)
    phi2 = result_data_phi.copy()
    elevation = 18.0
    dr = 100.0
    nbins = result_data_zh.shape[0]  # andersrum? erst times dann bins?
    times = result_data_zh.shape[1]
    toph, bottomh, topi, bottomi, bh = melting_layer_qvp(result_data_zh, result_data_zdr, result_data_rho,
                                                         elevation, dr, nbins, height_limit=10.)
    #diffml=np.zeros(times)
    #print("ML", topi, bottomi, phi2.shape)
    indexx=np.zeros(times)
    for i in range(times):
        #print("ML", int(topi[i] + 10), int(topi[i] + 20))
        #print("TIMES:", i, phi2[int(topi[i] + 10):int(topi[i] + 20), i])
        index = int(topi[i] + 10) + np.argmin(phi2[int(topi[i] + 10):int(topi[i] + 20), i])
        indexx[i]=index
        indexu = int(bottomi[i] - 10)
        diffml = phi2[index, i] - phi2[indexu, i]
        steps=index-indexu
        for l in range(steps):
            phi2[indexu+l, i]= phi2[indexu, i] + l*(diffml/steps)
    #modified kdp
    result_data_phi2 = phi2.copy()
    #vorher mit L=11, ersetzt
    #result_data_kdp2 = wrl.dp.kdp_from_phidp_convolution(result_data_phi2.T, L=21, dr=0.1).T
    #durch
    result_data_kdp2 = wrl.dp.kdp_from_phidp(result_data_phi2.T, winlen=21, dr=0.1).T
    ############################################################
    ##### Einschub Retrieval Alexander Ryzhkov
    zv=result_data_zh - result_data_zdr
    zdrlin = wrl.trafo.idecibel(result_data_zdr)
    zhlin= wrl.trafo.idecibel(result_data_zh)
    zvlin = wrl.trafo.idecibel(zv)
    zdp=zhlin-zvlin
    ratio=(zdp/result_data_kdp2/32.)**(1/2.)
    ratio_mask = np.where(ratio <= .15)
    kdp2_mask = np.where(result_data_kdp2 < 0.01)
    ratio[ratio_mask] = np.nan
    result_data_dm = -0.1 + 2.0*ratio
    gamma=0.78*ratio**2.
    result_data_dm[kdp2_mask] = np.nan
    result_data_logNt= 0.1* result_data_zh - 2.0 * np.log10(gamma)-1.33
    result_data_logNt[kdp2_mask] = np.nan
    #print(result_data_logNt)

    result_data_logIWC = (4.0 * 10**(-3) * 32. * result_data_kdp2) / (1 - zdrlin ** (-1))
    result_data_logIWC[kdp2_mask] = np.nan
    result_data_logIWC = np.log10(result_data_logIWC)
    print('neues',result_data_logIWC)

    #noch nicht-pol Variante von Hogan et al.
    # result_data_logIWC_Z = 0.06 * result_data_zh - 0.02 * temp_new - 1.7
    # #
    # for i in range(times):
    #     result_data_dm[0:int(indexx[i]),i]=np.nan
    #     result_data_logIWC[0:int(indexx[i]), i] = np.nan
    #     result_data_logNt[0:int(indexx[i]), i] = np.nan
    #     result_data_logIWC_Z[0:int(indexx[i]), i] = np.nan


    #print(result_data_logIWC)
    ############################################################
    #COSMO prozessing fuer out_2013-03-01-00 bis out_2014-07-07-00
    #cosmo_path = '/automount/cluma04/CNRW/CNRW_4.23/cosmooutput/' \
    #             'out_{0}-00/'.format(date)

    # COSMO prozessing fuer out_2014-08-05-00 bis jetzt
    #cosmo_path = '/automount/cluma04/CNRW/CNRW_5.00_grb2/cosmooutput/' \
    #             'out_{0}-00/'.format(date)

    # COSMO prozessing von Anna für die Statistik in ML Faelle
    cosmo_path = '/automount/cluma04/CNRW/CNRW5_output/' \
                 'out_{0}-00/'.format(date)
    #cosmo_path = '/home/silke/Python/testdata/QVP/' \
    #             'out_{0}-00/'.format(date)
    print("cosmopath", cosmo_path)
    is_cosmo = os.path.exists(cosmo_path)

    ## read grid from gribfile
    # filename = cosmo_path + 'lfff00{0}{1:02d}00'.format(16,0)
    # ll_grid = get_grid_from_gribfile(filename)

    if is_cosmo:
        # read grid from constants file
        filename = cosmo_path + 'lfff00000000c'
        print("filename",filename)
        rlat = mecc.get_ecc_value_from_file(filename, 'shortName', 'RLAT')
        rlon = mecc.get_ecc_value_from_file(filename, 'shortName', 'RLON')
        ll_grid = np.dstack((rlon, rlat))
        print("rlat, rlon", rlat.shape, rlon.shape)

        # calculate juxpol grid indices
        #juxpol_coords = (6.4569489, 50.9287272)
        #llx = np.searchsorted(ll_grid[0, :, 0], juxpol_coords[0], side='left')
        #lly = np.searchsorted(ll_grid[:, 0, 1], juxpol_coords[1], side='left')
        #print("Coords Juelich: ({0},{1})".format(llx, lly))

        # calculate boxpol grid indices
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
        tcount = int(divmod((end_time - start_time).total_seconds(), 60 * 30)[0] + 1)
        print(tcount)

        # create timestamps every full 30th minute (00, 30)
        cosmo_dt_arr = [(start_time + dt.timedelta(minutes=30 * i)) for i in range(tcount)]
        cosmo_time_arr = [(start_time + dt.timedelta(minutes=30 * i)).time() for i in range(tcount)]
        print(cosmo_dt_arr)

        # create temperature array and read from grib files
        temp = np.ones((len(cosmo_time_arr), 50)) * np.nan
        for it, t in enumerate(cosmo_time_arr):
            filename = cosmo_path + 'lfff00{:%H%M%S}'.format(t)
            print(filename)
            try:
                temp[it, ...] = mecc.get_ecc_value_from_file(filename, 'shortName', 't')[llx, lly, ...]
            except IOError:
                pass

        # we need degree celsius, not kelvins
        temp = temp - 273.15



        # text output temperature
        fn = '{0}_output_{1}_{2}.txt'.format('temp', location, date)

        header = "Temperature: {0}\tDATE: {1}\n".format(
            location, date)
        y_hhl = np.diff(hhl) / 2 + hhl[:-1]

        print(y_hhl.shape, temp)


        cosmotime = np.array([(ct - dt.datetime(1970, 1, 1)).total_seconds() for ct in cosmo_dt_arr])
        print(cosmotime.shape, hhl.shape, temp.shape)
        f = scipy.interpolate.interp2d(cosmotime, y_hhl, temp.T, kind='cubic')
        y = beam_height
        x = mdates.date2num(dt_src)
        temp_new = f(x, y)
        index, tindex = temp.T.shape
        cosmo_time_arr = [(start_time + dt.timedelta(minutes=30 * i)).time() for
                          i in range(tindex)]
        header2 = "\nBin Height/km " + ' '.join(
            ['{0:02d}:{1:02d}'.format(tx.hour, tx.minute) for tx in
             cosmo_time_arr]) + '\n'
        iarr = np.array(range(index))
        fmt = ['%d', '%0.4f'] + ['%0.4f'] * tindex
        print(iarr.shape, y_hhl.shape, temp.shape)
        np.savetxt(textfile_path + '/' + fn,
                   np.vstack([iarr, y_hhl[::-1], temp[:, ::-1]]).T,
                   fmt=fmt, delimiter=' ',
                   header=header + header2)

    result_data_logIWC_Z = 0.06 * result_data_zh - 0.02 * temp_new - 1.7
    #
    for i in range(times):
        result_data_dm[0:int(indexx[i]), i] = np.nan
        result_data_logIWC[0:int(indexx[i]), i] = np.nan
        result_data_logNt[0:int(indexx[i]), i] = np.nan
        result_data_logIWC_Z[0:int(indexx[i]), i] = np.nan

    # -----------------------------------------------------------------
    # result_data_... plotting

    # time stamps for plotting
    dt_start = mdates.date2num(dt_src[0])
    dt_stop = mdates.date2num(dt_src[-1])

    # contour lines
    # if true plot contours, if false plot pcolormesh
    contour = True

    # zh Contourlevels for zh isoline overlay
    contourlevels = [0, 5, 10, 20, 30]

    # define the colors
    colors = '#00f8f8', '#01b8fb', '#0000fa', '#00ff00', '#00d000', '#009400', \
             '#ffff00', '#f0cc00', '#ff9e00', '#ff0000', '#e40000', '#cc0000', \
             '#ff00ff', '#a06cd5'

    # dictionaries for moments, with some configuration
    zdr = {'name': 'zdr',
           'data': result_data_zdr,
           'levels': [-2, -1, 0, .1, .2, .3, .4, .5, .6, .8, 1.0, 1.2, 1.5,
                      2.0, 2.5, 3.0],
           'title': '$\mathrm{\mathsf{Z_{DR}}}$',
           'cb_label': 'Differential Reflectivity (dB)'}

    zh = {'name': 'zh',
          'data': result_data_zh,
          'levels': [-10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
                     60, 65],
          'title': '$\mathrm{\mathsf{Z_{H}}}$',
          'cb_label': 'Reflectivity (dBz)'}

    phi = {'name': 'phi',
           'data': result_data_phi,
           'levels': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35,
                      40, 100],
           'title': '$\mathrm{\mathsf{\phi_{DP}}}$',
           'cb_label': 'Differential Phase (deg)'}

    phi2 = {'name': 'phi2',
           'data': result_data_phi2,
           'levels': [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35,
                      40, 100],
           'title': '$\mathrm{\mathsf{\phi_{DP}}}$',
           'cb_label': 'Differential Phase (deg)'}

    rho = {'name': 'rho',
           'data': result_data_rho,
           'levels': [.7, .8, .85, .9, .92, .94, .95, .96, .97, .98, .985,
                      .99, .995, 1, 1.1],
           'title': r'$\mathrm{\mathsf{\rho_{hv}}}$',
           'cb_label': 'Crosscorrelation Coefficient'}

    kdp = {'name': 'kdp',
           'data': result_data_kdp,
           'levels': [-1, -0.5, -0.1, 0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80,
                      1.0, 2., 3., 4.],
           'title': '$\mathrm{\mathsf{K_{DP}}}$',
           'cb_label': 'Specific differential Phase (deg/km)'}

    kdp2 = {'name': 'kdp2',
           'data': result_data_kdp2,
           'levels': [-1, -0.5, -0.1, 0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80,
                      1.0, 2., 3., 4.],
           'title': '$\mathrm{\mathsf{K_{DP}}}$',
           'cb_label': 'Specific differential Phase (deg/km)'}

    dm = {'name': 'dm',
            'data': result_data_dm,
            'levels': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
            'title': '$\mathrm{\mathsf{D_{m}}}$',
            'cb_label': 'Mean volume diameter (mm)'}

    logNt = {'name': 'logNt',
          'data': result_data_logNt,
          'levels': [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5],
          'title': '$\mathrm{\mathsf{lg(N_t)}}$',
          'cb_label': 'Log. conc. of ice particles (1/L)'}

    logIWC = {'name': 'logIWC',
             'data': result_data_logIWC,
             'levels': [-2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 1.0, 1.5, 2.0],
             'title': '$\mathrm{\mathsf{lg(IWC)}}$',
             'cb_label': 'Log. ice water content ($\mathrm{\mathsf{g/m^3}}$)'}

    logIWC_Z = {'name': 'logIWC_Z',
              'data': result_data_logIWC_Z,
              'levels': [-2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4],
              'title': '$\mathrm{\mathsf{lg(IWC(Z))}}$',
              'cb_label': 'Log. ice water content a la Hogan ($\mathrm{\mathsf{g/m^3}}$)'}


    moments = {'zh': zh,
               'zdr': zdr,
               'phi': phi,
               'rho': rho,
               'kdp': kdp,
               'kdp2': kdp2,
               'phi2': phi2,
               'dm': dm,
               'logNt': logNt,
               'logIWC': logIWC,
               'logIWC_Z': logIWC_Z
               }

    # x-y-grid for radar data
    y = beam_height
    x = mdates.date2num(dt_src)
    X, Y = np.meshgrid(x, y)

    if is_cosmo:
        # x-y-grid for grib data
        # we use hhl, so we have to calculate the mid of the layers.
        y_hhl = np.diff(hhl) / 2 + hhl[:-1]
        print(y_hhl.shape, hhl.shape)
        x_temp = mdates.date2num(cosmo_dt_arr)
        X1, Y1 = np.meshgrid(x_temp, y_hhl)

    # cfg dict
    cfg = {'dt_start': dt_start,
           'dt_stop': dt_stop,
           'contour': contour,
           'contourlevels': contourlevels,
           'colors': colors,
           'X': X,
           'Y': Y,
           'beam_height': beam_height
           }

    # iterate over moments dict
    for k, mom in moments.items():
        # create figure
        mom['fig'], mom['ax'] = fig_ax(mom['title'], plot_width, plot_height)
        # add zh overlay contour
        add_contour(mom['ax'], X, Y, moments['zh']['data'], contourlevels,
                    manual='True', origin='lower', colors='k', alpha=0.8,
                    linewidths=1,
                    extent=[dt_start, dt_stop, -0.11, beam_height[-1]])
        if is_cosmo:
            # add temperature contour
            add_contour(mom['ax'], X1, Y1, temp.T, [-15, -10, -5, 0], manual='True',
                        origin='lower', colors='k', alpha=0.8,
                        linewidths=2)
        # add data to images
        add_plot(mom, cfg)
        # save images
        mom['fig'].savefig(
            plot_path + mom['name'] + '_' + location + '_' + str(round(elevation, 1)) + '_' + date + '.png',
            dpi=300, bbox_inches='tight')

        # text output
        fn = '{0}_output_{1}_{2}_{3}.txt'.format(mom['name'], location, str(round(elevation, 1)), date)

        header = "Radar: {0}\n\n\t{1}-DATA\tDATE: {2}\n".format(
            location, mom['name'].upper(), date)

        index, tindex = mom['data'].shape
        radar_time_arr = [(start_time + dt.timedelta(minutes=5 * i)).time() for
                          i in range(tindex)]
        header2 = "\nBin Height/km " + ' '.join(
            ['{0:02d}:{1:02d}'.format(tx.hour, tx.minute) for tx in
             radar_time_arr]) + '\n'
        iarr = np.array(range(index))
        fmt = ['%d', '%0.4f'] + ['%0.13f'] * tindex
        np.savetxt(textfile_path + '/' + fn,
                   np.vstack([iarr, beam_height, mom['data'].T]).T, fmt=fmt,
                   header=header + header2)

    plt.show()
    t2 = dt.datetime.now()
    print('Elapsed time: %12.7f seconds.' % ((t2 - t1).total_seconds()))

# =======================================================
if __name__ == '__main__':
    qvp_Boxpol()
