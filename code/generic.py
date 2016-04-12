# /usr/bin/env python2
""" Generic Tools Libary.
Contains:
    EnhancedObject      Base class that implements timing methods
"""

__author__ = "Sotiris Lamprinidis"
__copyright__ = "Copyright 2015, Sotiris Lamprinidis"
__credits__ = ["Sotiris Lamprinidis"]
__license__ = "GPL"
__version__ = "0.06"
__maintainer__ = "Sotiris Lamprinidis"
__email__ = "sot.lampr@gmail.com"
__status__ = "Testing"


class EnhancedObject(object):
    '''A Base class that implements a progress monitoring method
    Usage:
    EnhancedObject.progress(minimum, maximum)
    '''
    def __init__(self):
        pass

    @staticmethod
    def progress(mini, maxi, var_to_update):
        """ Calculate human-readable form of progress """
        progress = (mini/float(maxi-1))*100
        var_to_update.set(progress)
