# /usr/bin/env python2

"""Generic Audio Classifier - Classification Library
Contains:

"""
from collections import OrderedDict

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit, cross_val_score
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Pylint doesn't handle well module imports
# pylint: disable=import-error
from generic import EnhancedObject
from toolbox import Yaafe
# pylint: enable = import-error


__author__ = "Sotiris Lamprinidis"
__copyright__ = "Copyright 2015, Sotiris Lamprinidis"
__credits__ = ["Sotiris Lamprinidis"]
__license__ = "GPL"
__version__ = "0.06"
__maintainer__ = "Sotiris Lamprinidis"
__email__ = "sot.lampr@gmail.com"
__status__ = "Testing"


# Names like x, y are commonly used symbols in machine learning task
# pylint: disable=invalid-name
class FeatureEvaluator(EnhancedObject):
    '''Evaluates individual Features and combines the K best of them. Designed
    to work with Database and Yaafe objects
    Methods:
        evaluate(n_folds=6)             Get the scoring of the individual
                                            features
        run()                           Show scores for ascending k

    Usage:
        FeatureEvaluator(db, yaafe)
    '''
    def __init__(self, db, yaafe):
        self.db = db
        self.features = yaafe._features.keys()
        self.yaafe = yaafe
        self.y = self.db.get_y()
        self.scores = list()

    def evaluate(self, progress_var, Main, n_iter=6):
        '''Populates a list with the scores for each feature'''
        ssp = StratifiedShuffleSplit(self.y, n_iter=n_iter)
        for i_feat, feat in enumerate(self.features):
            self.progress(i_feat, len(self.features), progress_var)
            X = self.db.get_X(feat)
            X_temp = \
                np.nan_to_num(StatisticalTransformer().fit_transform(X))
            pipe = Pipeline([
                ('pca_lda', PCA_LDA()),
                ('log', LogisticRegression())
                ])
            score = np.mean(cross_val_score(pipe, X_temp, self.y, cv=ssp))
            self.scores.append([score, i_feat, feat])
            Main.update_idletasks()
        self.scores.sort(reverse=True)

    def get_K_best_master(self, K, entries=None, y=None):
        '''Get K best features and their selectors'''
        entries = self.db.entries if entries is None else entries
        y = self.y if y is None else y
        X_gold, selectors = [list() for i in range(0, 2)]
        K = len(self.scores) if K == 0 else K
        for k in range(0, K):
            X, _, _ = self._get_full_k(k, entries)
            X_temp, selector = self._process(X, y)
            selectors.append(selector)
            X_gold.append(X_temp)
        return np.column_stack((X_gold)), selectors

    def get_K_best_slave(self, K, selectors, entries=None):
        '''Get K Best Features'''
        entries = self.db.entries if entries is None else entries
        X_gold = list()
        K = len(self.scores) if K == 0 else K
        for k in range(0, K):
            X, _, _ = self._get_full_k(k, entries)
            X_temp = self._process(X, selectors[k])
            X_gold.append(X_temp)
        return np.column_stack((X_gold))

    def _get_full_k(self, k, entries):
        i_feat, feat = self._get_feat_k(k)
        X = self.db.get_X(feat, entries)
        return X, i_feat, feat

    def _get_feat_k(self, k):
        i_feat, feat = self.scores[k][1], self.scores[k][2]
        return i_feat, feat

    @staticmethod
    def _process(X, wild):
        """ preprocess a sample.
            wild: could be either a targets vector
                    -> returns X_transformed, transformer_item
                  or a transformer, where we apply the transformation
                    -> returns X_transformed
        """

        X_temp = np.nan_to_num(StatisticalTransformer().fit_transform(X))
        if isinstance(wild, np.ndarray):
            selector = PCA_LDA().fit(X_temp, wild)
            X_temp_alt = selector.transform(X_temp)
            return X_temp_alt, selector
        else:
            X_temp_alt = wild.transform(X_temp)
            return X_temp_alt

    def get_K_feature_plan(self, K):
        """ Get the feature plan for selected K best features"""
        new_feat_plan = list()
        for k in range(0, K):
            _, feat = self._get_feat_k(k)
            new_feat_plan.append([feat, self.yaafe._features[feat]])

        return dict(new_feat_plan)

    def run(self, K, Main, var_to_update, n_iter=6):
        """ Run an overview evaluation session for incrementing
            aggregates of features
        """
        for k in range(1, K):
            self.progress(k, K, var_to_update)
            y_true, y_pred, _ = self.run_k(k, n_iter=n_iter)
            Main.evaluation_results.insert(
                'end', "{} best: {}\n".format(
                    k, round(accuracy_score(y_true, y_pred), 3)))
            Main.update_idletasks()

    def run_k(self, k, n_iter):
        """ Run a specific combination of features to get precise
            cross-validated results
        """
        y_true, y_pred = [list() for i in range(0, 2)]
        ssp = StratifiedShuffleSplit(self.y, n_iter=n_iter, test_size=0.2)
        for train, test in ssp:
            entries_train, entries_test, y_train, y_test =\
                self.db.entries[train], self.db.entries[test],\
                self.y[train], self.y[test]
            X_train, selectors = self.get_K_best_master(
                k, entries_train, y_train)
            X_test = self.get_K_best_slave(k, selectors, entries_test)
            y_true.extend(y_test)
            clf = LogisticRegression()
            y_pred.extend(clf.fit(X_train, y_train).predict(X_test))
        return y_true, y_pred, len(X_test[0])

    def packer(self, K):
        '''Gathers all necessary elements for future classifications'''
        feature_plan_ = self.get_K_feature_plan(K)
        target_names = self.db.subdirs
        X_gold, selectors = [list() for i in range(0, 2)]
        feature_plan = OrderedDict()
        for key, val in feature_plan_.items():
            feature_plan[key] = val
        for feat in feature_plan.keys():
            X = self.db.get_X(feat)
            X_temp = np.nan_to_num(StatisticalTransformer().fit_transform(X))
            selector = PCA_LDA().fit(X_temp, self.y)
            X_temp_alt = selector.transform(X_temp)
            selectors.append(selector)
            X_gold.append(X_temp_alt)
        X_gold = np.column_stack((X_gold))
        estimator = LogisticRegression().fit(X_gold, self.y)
        return list([feature_plan, target_names, selectors, estimator])


class PCA_LDA(BaseEstimator, TransformerMixin):
    '''PCA and LDA Feature union'''
    def __init__(self):
        self.pipe = None    # Placeholder

    def fit(self, X, y):
        """ Fit the data """
        n_len = len(X[0])
        if n_len > 30:
            n_cut = 13
        elif n_len > 15:
            n_cut = 7
        else:
            n_cut = 3
        uni = FeatureUnion([
            ('lda', LDA(n_components=n_cut-1)),
            ('pca', PCA(n_components=n_cut))])
        pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ('union', uni)])
        self.pipe = pipe
        self.pipe.fit(X, y)
        return self

    def transform(self, X):
        """ Transform a matrix X """
        X_new = self.pipe.transform(X)
        return X_new


class StatisticalTransformer(BaseEstimator, TransformerMixin):
    '''Transform input n dimension arrays in single vector of statistical
        descriptors
    Usage:
    stat = StatisticalTransformer()
    X_transformed = stat.fit_transform(X)
    '''
    def __init__(self, onebone=True):
        pass

    def fit(self, *_):
        """ Dummy method for compatibility with scikit-learn """
        return self

    @staticmethod
    def transform(X, **_):
        """ Take some statistical measures and transform the data """
        X_temp = list()
        for entry in X:
            mn = np.amin(entry, axis=0)
            sd = np.std(entry, axis=0)
            av = np.mean(entry, axis=0)
            mx = np.amax(entry, axis=0)
            X_temp.append(np.append([mn, sd], [av, mx]))
        return np.array(X_temp)


def query(audio_file, sample_rate, package):
    '''Reads an audio file and predicts its class'''
    feature_plan, target_names, selectors, estimator = package
    yaafe = Yaafe(sample_rate, feature_plan)
    data = yaafe.process(audio_file)
    X_gold = list()
    for i_feat, feat in enumerate(feature_plan.keys()):
        X = data[feat]
        X.shape = (1, X.shape[0], X.shape[1])
        X_temp = np.nan_to_num(StatisticalTransformer().fit_transform(X))
        # X_temp.shape = X_temp.shape[1]
        X_temp_alt = selectors[i_feat].transform(X_temp)
        X_gold.append(X_temp_alt)
    X_gold = np.column_stack((X_gold))
    y_pred = estimator.predict(X_gold)
    y_proba = estimator.predict_proba(X_gold)
    proba = "{}%".format(round(y_proba[0][y_pred], 2))
    label = target_names[y_pred[0]]
    return y_pred, proba, label
