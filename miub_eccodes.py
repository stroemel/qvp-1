# coding=utf-8
#------------------------------------------------------------------------------
# Name:        miub_eccodes.py
# Purpose:     miub eccodes functions
#
# Author:      Kai Muehlbauer
#
# Created:     23.03.2016
# Copyright:   (c) Kai Muehlbauer 2016
# Licence:     The MIT License
#------------------------------------------------------------------------------
# This module is far from being mature
# may change without notice

import numpy as np
import eccodes as ecc
#from collections import OrderedDict
#from dictdiffer import diff

def get_ecc_value_from_file(filename, keyname, keyvalue):

    f = open(filename)

    gids = get_ecc_gids(filename)
    value = get_ecc_value(gids, keyname, keyvalue)
    release_ecc_gids(gids)

    f.close()

    return value
    
def get_ecc_value(gids, keyname, keyvalue):
    
    first = True
    
    for i, gid in enumerate(gids):
        if ecc.codes_get(gid, keyname) == keyvalue:
            if first:            
                Ni = ecc.codes_get(gid, 'Ni')
                Nj = ecc.codes_get(gid, 'Nj')
                data = np.array(ecc.codes_get_values(gid))
                first = False
            else:
                data = np.dstack((data, np.array(ecc.codes_get_values(gid))))

    return np.squeeze(data.reshape(Ni, Nj, -1))

def get_ecc_subkey(gids, keyname, keyvalue, subkey):
    
    for i, gid in enumerate(gids):
        if ecc.grib_get(gid, keyname) == keyvalue:
                data = ecc.grib_get(gid, subkey)
                break
    
    return data

def get_ecc_gids(filename):
    f = open(filename)
    msg_count = ecc.codes_count_in_file(f)
    # TODO: this is only for grib files
    gid_list = [ecc.codes_grib_new_from_file(f) for i in range(msg_count)]
    #print(gid_list)
    f.close()
    return gid_list

def release_ecc_gids(gids):
    for gid in gids:
        ecc.codes_release(gid)

def get_ecc_data(gids):
    
    grib_dict = {}
    
    for i, gid in enumerate(gids):
        sn = ecc.grib_get(gid, 'short_name')
        if sn not in grib_dict:
            tmp = {}
            Ni = ecc.grib_get(gid, 'Ni')
            Nj = ecc.grib_get(gid, 'Nj')
            tmp['data'] = np.array(ecc.grib_get_values(gid)).reshape(Nj,Ni,1)
            tmp['shape'] = tmp['data'].shape            
            grib_dict[sn] = tmp
        else:
            #print(grib_dict[sn]['data'].shape)
            #if ecc.grib_is_defined(gid, 'levels'):
            #    print(ecc.grib_get(gid, 'levels'))
            Ni, Nj, Nk = grib_dict[sn]['data'].shape
            grib_dict[sn]['data'] = np.dstack((grib_dict[sn]['data'], np.array(ecc.grib_get_values(gid)).reshape(Nj,Ni)))
            grib_dict[sn]['shape'] = grib_dict[sn]['data'].shape                
    
    return grib_dict


def get_ecc_variable(filename, key, namespace=None, skipkeys=None):
    
    gids = get_ecc_gids(filename)    
    
    grib_dict = {}
    
    for i, gid in enumerate(gids):
        tmp = get_ecc_msg(gid, namespace=namespace, skipkeys=skipkeys)
        sn = tmp['shortName']
        print("Shortname:", sn)
        for k, v in tmp.items():
            print(k, type(v))
        if sn == key:
            if sn not in grib_dict: 
                grib_dict[sn] = tmp
            #else:
            #    d = list(diff(grib_dict[sn], tmp))
            #    print(d)
            #    for key in d:
            #        if key[0] == 'change':
            #            try:
            #                grib_dict[sn][key[1]] = np.dstack((grib_dict[sn][key[1]], tmp[key[1]]))
            #            except ValueError:
            #                grib_dict[sn][key[1]] = np.array([grib_dict[sn][key[1]], tmp[key[1]]])
    return grib_dict

def get_ecc_file(filename, namespace=None, skipkeys=None):
    
    gid_list = get_ecc_gids(filename)

    print(len(gid_list), gid_list)

    grib_dict = {}

    for i, gid in enumerate(gid_list):
        tmp = get_ecc_msg(gid, namespace=namespace, skipkeys=skipkeys)
        sn = tmp['shortName']
        if sn not in grib_dict:
            grib_dict[sn] = tmp
        else:
            d = list(diff(grib_dict[sn], tmp))
            for key in d:
                if key[0] == 'change':
                    try:
                        grib_dict[sn][key[1]] = np.dstack((grib_dict[sn][key[1]], tmp[key[1]]))
                    except ValueError:
                        grib_dict[sn][key[1]] = np.array([grib_dict[sn][key[1]], tmp[key[1]]])
    return grib_dict
    
def get_ecc_msg_keys(gid, namespace=None, skipkeys=None):
    """Retrieve keys from one particular ecc message

    Parameters
    ----------
    gid : ecc message id
    namespace : string
        namespace to be retrieved, defaults to None (means all)
        'ls', 'parameter', 'time', 'geography', 'vertical', 'statistics', 'mars'
    skipkeys  : list of strings
        keys to be skipped, defaults to None
        possible keys: 'computed', 'coded', 'edition', 'duplicates', 'read_only', 'function'
        
    Returns
    -------
    data : list of ecc message keys 
    """
    
    # get key iterator
    iterid = ecc.codes_keys_iterator_new(gid, namespace)
    
    # Different types of keys can be skipped
    if skipkeys:
        if 'computed' in skipkeys:
            ecc.codes_skip_computed(iterid)
        if 'coded' in skipkeys:
            ecc.codes_skip_coded(iterid)
        if 'edition' in skipkeys:
            ecc.codes_skip_edition_specific(iterid)
        if 'duplicates' in skipkeys:
            ecc.codes_skip_duplicates(iterid)
        if 'read_only' in skipkeys:
            ecc.codes_skip_read_only(iterid)
        if 'function' in skipkeys:    
            ecc.codes_skip_function(iterid)
    
    data = []
    # iterate over message keys
    while ecc.codes_keys_iterator_next(iterid):
        keyname = ecc.codes_keys_iterator_get_name(iterid)
        # add keyname-keyvalue-pair to output dictionary
        data.append(keyname)
    
    # release iterator
    ecc.codes_keys_iterator_delete(iterid)

    return data

        
def get_ecc_msg(gid, namespace=None, skipkeys=None):
    """Read data from one particular ecc message

    Parameters
    ----------
    gid : ecc message id
    namespace : string
        namespace to be retrieved, defaults to None (means all)
        'ls', 'parameter', 'time', 'geography', 'vertical', 'statistics', 'mars'
    skipkeys  : list of strings
        keys to be skipped, defaults to None
        possible keys: 'computed', 'coded', 'edition', 'duplicates', 'read_only', 'function'
        

    Returns
    -------
    data : dictionary of ecc message contents 
    """
    
    # get key iterator
    iterid = ecc.codes_keys_iterator_new(gid, namespace)

    # Different types of keys can be skipped
    if skipkeys:
        if 'computed' in skipkeys:
            ecc.codes_skip_computed(iterid)
        if 'coded' in skipkeys:
            ecc.codes_skip_coded(iterid)
        if 'edition' in skipkeys:
            ecc.codes_skip_edition_specific(iterid)
        if 'duplicates' in skipkeys:
            ecc.codes_skip_duplicates(iterid)
        if 'read_only' in skipkeys:
            ecc.codes_skip_read_only(iterid)
        if 'function' in skipkeys:    
            ecc.codes_skip_function(iterid)
    
    data = OrderedDict()
   
   # iterate over message keys
    while ecc.codes_keys_iterator_next(iterid):


        keyname = ecc.codes_keys_iterator_get_name(iterid)
        #print(keyname)


        #print("Size:", ecc.codes_get_size(gid, keyname))
        #print("Values:", ecc.codes_get_values(gid, keyname))
        #print("Array:", ecc.codes_get_values(gid, keyname))

        # try to get key values,
        # use get_array for sizes > 1 and get for sizes == 1
        if ecc.codes_get_size(gid,keyname) > 1:
            #print("has array", type is str)
            #print(type is not <type 'str'>)
            if ecc.codes_get_native_type(iterid, keyname) is not str:
                keyval = ecc.codes_get_array(gid, keyname, None)
            else:
                keyval = ecc.codes_get(gid, keyname, None)
            #print("Arr:", keyval)
        else:
            # Todo: fix reading mybits
            if keyname not in ['mybits']:
                keyval = ecc.codes_get(gid, keyname, None)
                #print("Val:", keyval)
            else:
                keyval = 'err'

        # add keyname-keyvalue-pair to output dictionary
        data[keyname] = keyval

    #print('Message processed')
    # release iterator
    ecc.codes_keys_iterator_delete(iterid)

    return data

if __name__ == '__main__':
    print('miub: Calling module <miub_eccodes> as main...')