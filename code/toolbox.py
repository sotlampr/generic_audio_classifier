#!/usr/bin/env python2

""" Generic Audio Classifier - Toolbox
Contatins:
    Database
    Yaafe
"""

import os
from collections import defaultdict
from glob import glob
import pickle

import numpy as np
from yaafelib import FeaturePlan, Engine, AudioFileProcessor

# Names like x, y are commonly used symbols in machine learning task
# pylint: disable=import-error
from generic import EnhancedObject
# pylint: enable=import-error


__author__ = "Sotiris Lamprinidis"
__copyright__ = "Copyright 2015, Sotiris Lamprinidis"
__credits__ = ["Sotiris Lamprinidis"]
__license__ = "GPL"
__version__ = "0.06"
__maintainer__ = "Sotiris Lamprinidis"
__email__ = "sot.lampr@gmail.com"
__status__ = "Testing"

# In machine learning tasks, variable names like x and y are common
# pylint: disable=invalid-name
class Database(EnhancedObject):
    '''An Indexed File Database that also holds processed Data.

    Attributes:
        base_dir        The base directory in which to work
        __extensions    The allowed extensions
        subdirs         Dictionary of Subdirectories as in ({i: 'dir'})
        entries         Full filenames list with path and subdirectory
                            as in ([filename, i])
        data            The Processed Data

    Functions:
        flush()             Flush the Database
        populate()          Populate subdirs and entries
        process(processor)  Process and gather the Data
        save(name)          Savesthe database to disk
        load(name)          Load a saved database
        get_X(feature)      Return data for specific features
        get_y()             Return the class indices vector
    Init:
        Database(base_dir, file_ext)
    '''

    __extensions = ['.wav', '.mp3', '.ogg', '.aiff', '.flac']

    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.flush()

    def flush(self):
        """ Initialize an empty database """
        self.subdirs = defaultdict()
        self.entries = []
        self.data = []
        return self

    def populate(self):
        """ Read directories and files into the dataase """
        self.subdirs_populate()
        self.entries_populate()
        return self

    def subdirs_populate(self):
        """ Read available sudirectories"""
        for i, subdir in enumerate(next(os.walk(self.base_dir))[1]):
            self.subdirs[i] = subdir

    def entries_populate(self):
        """ Glob and discover files """
        file_index = 0
        for i in self.subdirs:
            for filename in glob(os.path.join(
                    self.base_dir, self.subdirs[i], '*')):
                if os.path.splitext(filename)[1] in self.__extensions:
                    # This is an audio file
                    tmp = []
                    tmp.append(file_index)
                    tmp.append(filename)
                    tmp.append(i)
                    self.entries.append(tmp)
                    file_index += 1
                else:
                    # This is not an audio file
                    pass

    def process(self, processor, progress_var, Main):
        """ Process files with a processor function """
        data = []
        n_entries = len(self.entries)
        for i, entry, index in self.entries:
            self.progress(i, n_entries, progress_var)
            tmp = []
            tmp.append(i)
            tmp.append(entry)
            tmp.append(index)
            tmp.append(processor.process(entry))
            data.append(tmp)
            Main.update_idletasks()
        self.entries = np.array(data)
        return self

    def save(self, name):
        """ Save database in a pickle file """
        np.save(name, self.entries)
        out = open('{}_classes.pkl'.format(name), 'wb')
        pickle.dump(self.subdirs, out)
        out.close()
        return self

    def load(self, name):
        """ Load database from pickle file """
        self.entries = np.load('{}.npy'.format(name))
        classes = open('{}_classes.pkl'.format(name), 'rb')
        self.subdirs = pickle.load(classes)
        classes.close()
        return self

    def get_X(self, feat, entries=None):
        """ Retrieve the data in a numpy array """
        entries = self.entries if entries is None else entries
        X = []
        for _, _, _, data in entries:
            X.append(data[feat])
        return np.array(X)

    def get_y(self, entries=None):
        """ Retrieve the labels for the data"""
        entries = self.entries if entries is None else entries
        y = []
        for _, _, index, _ in entries:
            y.append(index)
        return np.array(y)


class Yaafe(EnhancedObject):
    '''Yaafe toolbox wrapper. To be used with Database object.

    Attributes:
        sample_rate     The Files' sample rate
        plan_filename   The Featue Plan filename

    Methods:
        process(audiofile)          Process audiofile and return features
        get_X(entries_list, feat)   Fetch array of processed data from Database
        get_y                       Fetch subdir i's from Database
    `
    Init:
        Yaafe(sample_rate, feature_plan)
    '''
    _features = {
        'spec_rolloff': ("SpectralRolloff blockSize=512 stepSize=128 "
                         "> StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'spec_shape': ("SpectralShapeStatistics blockSize=512 stepSize=128 "
                       "> StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'spec_flux': ("SpectralFlux blockSize=512 stepSize=128 >"
                      "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'amp_mod': ("AmplitudeModulation blockSize=512 stepSize=128 >"
                    "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'auto': ("AutoCorrelation  blockSize=512 stepSize=128 >"
                 "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'lpc': ("LPC  blockSize=512 stepSize=128 > StatisticalIntegrator "
                "NbFrames=40 StepNbFrames=8"),
        'loudness': ("Loudness blockSize=512 stepSize=128 >"
                     "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'mfcc': ("MFCC blockSize=512 stepSize=128 > StatisticalIntegrator "
                 "NbFrames=40 StepNbFrames=8"),
        'mel_spectrum': ("MelSpectrum blockSize=512, stepSize=128 >"
                         "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'obsi': ("OBSI blockSize=512 stepSize=128 > StatisticalIntegrator "
                 "NbFrames=40 StepNbFrames=8"),
        'obsir': ("OBSIR blockSize=512 stepSize=128 >"
                  "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'perc_sharp': ("PerceptualSharpness blockSize=512 stepSize=128 >"
                       "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'perc_spread': ("PerceptualSpread blockSize=512 stepSize=128 >"
                        "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'spect_crest': ("SpectralCrestFactorPerBand blockSize=512 "
                        "stepSize=128 > StatisticalIntegrator NbFrames=40 "
                        "StepNbFrames=8"),
        'spec_decr': ("SpectralDecrease blockSize=512 stepSize=128 >"
                      "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'spect_flat': ("SpectralFlatness blockSize=512 stepSize=128 >"
                       "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'spect_flat_band': ("SpectralFlatnessPerBand blockSize=512 "
                            "stepSize=128 > StatisticalIntegrator NbFrames=40 "
                            "StepNbFrames=8"),
        'spect_slope': ("SpectralSlope blockSize=512 stepSize=128 >"
                        "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'spect_var': ("SpectralVariation blockSize=512 stepSize=128 >"
                      "StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'temp_shape': ("TemporalShapeStatistics blockSize=512 stepSize=128 "
                       "> StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'zcr': ("ZCR blockSize=512 stepSize=128 > StatisticalIntegrator "
                "NbFrames=40 StepNbFrames=8"),
        'env_shape': ("EnvelopeShapeStatistics blockSize=512 stepSize=128"
                      " > StatisticalIntegrator NbFrames=40 StepNbFrames=8"),
        'comp_onest': ("ComplexDomainOnsetDetection blockSize=512 "
                       "stepSize=128 > StatisticalIntegrator NbFrames=40 "
                       "StepNbFrames=8"),
        }

    def __init__(self, sample_rate, features=None):
        if features is None:
            features = self._features
        self.sample_rate = sample_rate
        self.initialize(features)
        self.features = features

    def initialize(self, feature_dict):
        """ Run the required boilerplate for yaafe """
        self.feature_dict = feature_dict
        self.fp = FeaturePlan(
            sample_rate=self.sample_rate,
            normalize=0.98)
        for name, desc in self.feature_dict.items():
            self.fp.addFeature("{0}: {1}".format(name, desc))
        self.df = self.fp.getDataFlow()
        self.engine = Engine()
        self.engine.load(self.df)
        self.afp = AudioFileProcessor()
        return self

    def save_fplan(self, name):
        """ Save a feature plan (text file) """
        text_file = open("{}.txt".format(name), 'w')
        for name, desc in self.features.items():
            text_file.write("{}: {}".format(name, desc))
        text_file.close()

    def process(self, audiofile):
        """ Process function for running a file through yaafe's
            feature extractor
        """
        self.afp.processFile(self.engine, audiofile)
        out = self.engine.readAllOutputs()
        self.engine.flush()
        return sorted(out)
