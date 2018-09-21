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

import psutil
process = psutil.Process(os.getpid())

import warnings
warnings.filterwarnings('ignore')

"""
-----------------------------------------------------------------
 global data    
-----------------------------------------------------------------
"""
# this defines start and end time
# need to be within the same day
start_time = dt.datetime(2015, 8, 27, 13, 00)
end_time = dt.datetime(2015, 8, 27, 23, 00)

date = '{0}-{1:02d}-{2:02d}'.format(start_time.year, start_time.month, start_time.day)
#location = 'Bonn'
location = 'Essen'
#radar_path='/automount/radar/scans/{0}/{0}-{1:02}/{2}'.format(start_time.year, start_time.month, date)
#radar_path='/home/silke/data/BlitzeisBerlin/Revision'#.format(start_time.year, start_time.month, date)
#radar_path='/home/silke/data/Climatology/20150108_ess'
radar_path='/home/silke/data/Climatology/20150827_ess'
output_path = '/home/silke/Python/output/Essen'
# choose scan
file_path = radar_path + '/' #+ 'n_ppi_280deg/'
textfile_path = output_path + '/{0}/textfiles/'.format(date)
plot_path = output_path + '/{0}/plots/'.format(date)

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

offset_z = 0#3
#offset_phi am 20150827 ist -140
offset_phi = -140#90
offset_zdr = 0.0#0.8
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



class rdata(object):
    def __init__(self):
        self._zh = None
        self._phi = None
        self._rho = None
        self._zdr = None
        self._vel = None
        self._radar_height = None
        self._range_samples = None
        self._range_step = None
        self._bin_count = None
        self._elevation = None
        self._date = None

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


class boxpol(rdata):
    """
    reads data from hdf5-file
    gelesen werden die Daten zh,phi,rho,zdr mit Zeitstempel.
    Mit der Rückgabe von no_file=0 wird zum Ausdruck gebracht, dass
    die Datei korrekt gelesen werden konnte.

    Im Fehlerfall (Rückgabe n0_file=1) werden Dummy-Daten returniert und
    auf den Bildschirm der Name der nicht gefundenen Datei ausgegeben.
    """
    def __init__(self, filename, **kwargs):
        super(boxpol, self).__init__()
        data = read_generic_hdf5(filename)
        #data = wrl.io.read_OPERA_hdf5(filename)
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


def get_moment(data, num):
    mom = data['dataset1/data{0}/data'.format(num)]
    nodata = data['dataset1/data{0}/what'.format(num)]['nodata']
    gain = data['dataset1/data{0}/what'.format(num)]['gain']
    offset = data['dataset1/data{0}/what'.format(num)]['offset']
    #mom = mom * gain + offset
    ret = np.ma.masked_equal(mom, nodata)
    ret = ret * gain + offset
    return ret

class dwdpol(rdata):
    """
    reads data from hdf5-file
    gelesen werden die Daten zh,phi,rho,zdr mit Zeitstempel.
    Mit der Rückgabe von no_file=0 wird zum Ausdruck gebracht, dass
    die Datei korrekt gelesen werden konnte.

    Im Fehlerfall (Rückgabe n0_file=1) werden Dummy-Daten returniert und
    auf den Bildschirm der Name der nicht gefundenen Datei ausgegeben.
    """
    def __init__(self, filename, **kwargs):
        super(dwdpol, self).__init__()
        data = wrl.io.read_OPERA_hdf5(filename)
        #print(data)
        #print(data['where']['height'])
        #print(data['what'])
        #exit(9)
        if data is not None:
            self._zh = get_moment(data, 1)
            self._phi = get_moment(data, 17)
            self._rho = get_moment(data, 19)
            self._zdr = get_moment(data, 5)
            self._radar_height = data['where']['height']
            self._range_samples = 1
            self._range_step = data['dataset1/where']['rscale']
            self._bin_count = data['dataset1/where']['nbins']
            self._elevation = data['dataset1/where']['elangle']

            self._date = dt.datetime.strptime(data['what']['date']+ data['what']['time'], '%Y%m%d%H%M%S')

def fig_ax(title, w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    return fig, ax

def add_contour(ax, X, Y, data, levels, **kwargs):
    cs = ax.contour(X, Y, data, levels, **kwargs)
    ax.clabel(cs, fmt='%2.0f', inline=True, fontsize=10)

def add_plot(mom, cfg):
    ax = mom['ax']
    levels = mom['levels']
    if cfg['contour']:
        im = ax.contourf(cfg['X'], cfg['Y'],mom['data'], levels=levels,
                         colors=cfg['colors'], origin='lower', axis='equal',
                         extent=[cfg['dt_start'], cfg['dt_stop'],
                                 -0.11, cfg['beam_height'][-1]])
    else:
        cmap = ListedColormap(cfg['colors'])  # , name='')
        norm = BoundaryNorm(levels, ncolors=cmap.N)#, clip=True)
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
    filetempl = 'mvol*0008'
    file_names = sorted(glob.glob(os.path.join(file_path, filetempl)))

    print(file_names[0])

    # get some specifics of the radar data
    ds0 = dwdpol(file_names[0])
    bin_range = ds0.range_samples * ds0.range_step
    dr = bin_range / 1000.
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
        file_names = sorted(glob.glob(os.path.join(file_path, filetempl)))
        print("FNAMES:", file_names)

        file_list = []
        for fname in file_names:
            print(os.path.splitext(os.path.basename(fname))[0].split('_')[1])
            time = dt.datetime.strptime(os.path.splitext(os.path.basename(fname))[0].split('_')[1], "%Y%m%d%H%M%S")
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

            dsl = dwdpol(fname)

            print(dsl.date)
            print("ZH:", dsl.zh.shape)
            print("RES:", result_data_zh.shape)
            print("---------> Reading finished")
            print("range_bin_dist", range_bin_dist)

            # add the offset
            dsl.zh += offset_z
            dsl.phi += offset_phi
            dsl.zdr += offset_zdr

            # noise reduction for rho_hv
            # vorher -23
            # vorher -21
            #noise_level = -26
            noise_level = -27

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

        #testplot
        wrl.vis.plot_cg_ppi(dsl.phi)
        plt.show()

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
        test = phi.copy()
        kdp = np.zeros_like(phi)
        #evtl Löschung Anfang
        for i, v in enumerate(phi):
            print("Timestep:", i)
            for j, w in enumerate(v):
                ma = movingaverage(w, 3, mode='same')
                test[i, j, :] = ma
                test[i, j, result_data_rho[i, j, :] < 0.7] = np.nan
                test[i, j, result_data_zh[i, j, :] < -10.] = np.nan
        #         # print(test[i, j, :])
        #         first = np.argmax(test[i, j, :] >= 0.)
        #         last = np.argmax(test[i, j, ::-1] >= 0)
        #         # print("f:", first, test[i, j, first], test[i, j, first-1])
        #         # print("l:", last, test[i, j, -last-1], test[i, j, -last])
        #         if first:
        #             test[i, j, :first + 1] = test[i, j, first]
        #         if last:
        #             test[i, j, -last:] = test[i, j, -last - 1]
        #evtl Löschung Ende

        #test[result_data_zh < -10] = np.nan
        # get kdp from phidp

        # V1: kdp from convolution, maximum speed
        kdp = wrl.dp.kdp_from_phidp_convolution(test, L=3, dr=1.0)

        # V2: fit ala ryzhkov, see function declared above
        #kdp = kdp_calc(test, wl=11)

        print(kdp.shape)

        # save data to file
        np.savez(output_path + '/' + location + '_' + str(round(elevation, 1)) + '_' + date + '_phidp_kdp', phi=phi, kdp=kdp, test=test)

    # median calculation
    print(result_data_rho)
    result_data_zh[result_data_rho <= 0.7] = np.nan
    result_data_zh[result_data_rho > 1.0] = np.nan
    result_data_zdr[result_data_rho <= 0.7] = np.nan
    result_data_zdr[result_data_rho > 1.0] = np.nan
    #result_data_phi[result_data_rho <= 0.7] = np.nan

    result_data_zh = np.nanmean(result_data_zh, axis=1).T
    result_data_zdr = np.nanmean(result_data_zdr, axis=1).T

    # mask kdp eventually,
    for i in range(360):
        k1 = kdp[:, i, :]
        k1[result_data_zh.T < -5.] = np.nan
        # print(k1.shape)
        kdp[:, i, :] = k1

    kdp[result_data_rho <= 0.7] = np.nan
    kdp[result_data_rho > 1.0] = np.nan
    result_data_kdp = np.nanmean(kdp, axis=1).T



    # mask phidp eventually,
    for i in range(360):
        k2 = test[:, i, :]
        k2[result_data_zh.T < -5.] = np.nan
        # print(k1.shape)
        test[:, i, :] = k2

    test[result_data_rho <= 0.7] = np.nan
    test[result_data_rho > 1.0] = np.nan
    result_data_phi = np.nanmean(test, axis=1).T

    result_data_rho[result_data_rho <= 0.7] = np.nan
    result_data_rho[result_data_rho > 1.0] = np.nan
    result_data_rho = np.nanmean(result_data_rho, axis=1).T

    print(np.nanmin(result_data_zdr), np.nanmax(result_data_zdr))
    result_data_zdr[result_data_zdr >= 20.0] = np.nan
    result_data_zdr[result_data_zdr <= -5] = np.nan
    print(np.nanmin(result_data_zdr), np.nanmax(result_data_zdr))
    # result_data_phi_median = stats.nanmean(phi, axis=1).T

    print("SHAPE1:", result_data_phi.shape, result_data_zdr.shape)
    plt.imshow(result_data_phi)
    plt.colorbar()
    plt.show()

    plt.plot(result_data_phi[:,60])
    plt.show()

    # -----------------------------------------------------------------
    # calulate beam_height: array with 350 elements
    # Elevation angle is hardcoded here
    # aendern
    #beam_height = (wrl.georef.beam_height_n(np.linspace(0, (bins - 1) * 100, bins), 28.0)
    #               + radar_height) / 1000
    print(range_bin_dist, round(elevation,1), radar_height)
    beam_height = (wrl.georef.beam_height_n(range_bin_dist, round(elevation,1))
                   + radar_height) / 1000

    print(beam_height)

    # # COSMO prozessing
    # cosmo_path = '/automount/cluma04/CNRW/CNRW_5.00_grb2/cosmooutput/' \
    #              'out_{0}-00/'.format(date)
    #
    # ## read grid from gribfile
    # #filename = cosmo_path + 'lfff00{0}{1:02d}00'.format(16,0)
    # #ll_grid = get_grid_from_gribfile(filename)
    #
    # # read grid from constants file
    # filename = cosmo_path + 'lfff00000000c'
    # print(filename)
    # rlat = mecc.get_ecc_value_from_file(filename, 'shortName', 'RLAT')
    # rlon = mecc.get_ecc_value_from_file(filename, 'shortName', 'RLON')
    # ll_grid = np.dstack((rlon, rlat))
    # print("rlat, rlon", rlat.shape, rlon.shape)
    #
    # # calculate boxpol grid indices
    # boxpol_coords = (7.071663, 50.73052)
    # llx = np.searchsorted(ll_grid[0, :, 0], boxpol_coords[0], side='left')
    # lly = np.searchsorted(ll_grid[:, 0, 1], boxpol_coords[1], side='left')
    # print("Coords Bonn: ({0},{1})".format(llx, lly))
    #
    # # read height layers from constants file
    # filename = cosmo_path + 'lfff00000000c'
    # hhl = mecc.get_ecc_value_from_file(filename,
    #                                    'shortName',
    #                                    'HHL')[llx, lly, ...]
    # # get km from meters
    # hhl = hhl / 1000.
    #
    # # reading temperature from associated comso files
    # # getting count of comso files
    # # beware only available from 00:00 to 21:30
    # tcount = int(divmod((end_time - start_time).total_seconds(), 60*30)[0] + 1)
    # print(tcount)
    #
    # # create timestamps every full 30th minute (00, 30)
    # cosmo_dt_arr = [(start_time + dt.timedelta(minutes=30 * i)) for i in range(tcount)]
    # cosmo_time_arr = [(start_time + dt.timedelta(minutes=30 * i)).time() for i in range(tcount)]
    # print(cosmo_dt_arr)
    #
    # # create temperature array and read from grib files
    # temp = np.zeros((len(cosmo_time_arr), 50))
    # for it, t in enumerate(cosmo_time_arr):
    #     filename = cosmo_path + 'lfff00{:%H%M%S}'.format(t)
    #     print(filename)
    #     temp[it, ...] = mecc.get_ecc_value_from_file(filename, 'shortName', 't')[llx,lly,...]
    #
    # # we need degree celsius, not kelvins
    # temp = temp - 273.15
    #
    # # text output temperature
    # fn = '{0}_output_{1}_{2}.txt'.format('temp', location, date)
    #
    # header = "Temperature: {0}\tDATE: {1}\n".format(
    #     location, date)
    # y_hhl = np.diff(hhl) / 2 + hhl[:-1]
    #
    # print(y_hhl.shape, temp)
    #
    # index, tindex = temp.T.shape
    # cosmo_time_arr = [(start_time + dt.timedelta(minutes=30 * i)).time() for
    #                   i in range(tindex)]
    # header2 = "\nBin Height/km " + ' '.join(
    #     ['{0:02d}:{1:02d}'.format(tx.hour, tx.minute) for tx in
    #      cosmo_time_arr]) + '\n'
    # iarr = np.array(range(index))
    # fmt = ['%d', '%0.4f'] + ['%0.4f'] * tindex
    # print(iarr.shape, y_hhl.shape, temp.shape)
    # np.savetxt(textfile_path + '/' + fn,
    #            np.vstack([iarr, y_hhl[::-1], temp[:,::-1]]).T,
    #            fmt=fmt, delimiter=' ',
    #            header=header + header2)


    # -----------------------------------------------------------------
    # result_data_... plotting

    # time stamps for plotting
    dt_start = mdates.date2num(dt_src[0])
    dt_stop = mdates.date2num(dt_src[-1])
    print(dt_src[0], dt_src[-1])
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
           #'levels': [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
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
           'levels': [-0.5, -0.1, 0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80,
                      1.0, 2., 3., 4.],
           'title': '$\mathrm{\mathsf{K_{DP}}}$',
           'cb_label': 'Specific differential Phase (deg/km)'}

    moments = {'zh': zh ,
               'zdr': zdr,
               'phi': phi,
               'rho': rho,
               'kdp': kdp
               }

    # x-y-grid for radar data
    y = beam_height
    x = mdates.date2num(dt_src)
    X, Y = np.meshgrid(x, y)

    # # x-y-grid for grib data
    # # we use hhl, so we have to calculate the mid of the layers.
    # y_hhl = np.diff(hhl) / 2 + hhl[:-1]
    # print(y_hhl.shape, hhl.shape)
    # x_temp = mdates.date2num(cosmo_dt_arr)
    # X1, Y1 = np.meshgrid(x_temp, y_hhl)

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
        # # add temperature contour
        # add_contour(mom['ax'], X1, Y1, temp.T, [-15, -10, -5, 0], manual='True',
        #         origin='lower', colors='k', alpha=0.8,
        #         linewidths=2)
        # add data to images
        add_plot(mom, cfg)
        # save images
        mom['fig'].savefig(plot_path + mom['name'] + '_' + location + '_' + str(round(elevation, 1)) + '_' + date + '.png',
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
