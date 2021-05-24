#! /usr/bin/python
# -*- coding: utf-8 -*-
# Copyright Fabrice Silva, 2012.

"""
Utilities
"""
import warnings

try:
    import h5py
    from . import hdf5pickle as pickle
    method = 'hdf5'
except ImportError:
    import six.moves.cPickle as pickle
    method = 'basic'

## Serialization
# TODO : add moreesc version in saved files.
def _pickle(obj, filename):
    " Serialize Profile. "
    if method == 'hdf5':
        pickle.dump(obj, filename)
    elif method == 'basic':
        with open(filename, 'wb') as fid:
            pickle.dump(obj, fid, pickle.HIGHEST_PROTOCOL)
    else:
        raise IOError()
    return


def _unpickle(filename):
    " Deserialize Profile. "
    if method == 'hdf5':
        from .hdf5pickle import load
        try:
            obj = load(filename)
        except OSError:
            # Occurs when h5py can not open file
            # (when it is not an HDF5 file for example)
            pass
        else:
            return obj

    import six.moves.cPickle as pickle
#    with open(filename, 'r', encoding='utf-8', errors='ignore') as fid:
    with open(filename, 'rb') as fid:
        obj = pickle.load(fid)
    return obj

__pickle_common_doc = """
Load data saved with the :meth:`%(class)s.save` method:

Parameters
----------
filename: file-like object (file or string)
  Name of the file to load.

Returns
-------
obj: %(output)s
  Stored %(class)s
"""
