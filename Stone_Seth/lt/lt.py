"""
A new methodology for long-term maintenance of
WiFi fingerprinting radio maps
R. Montoliu, E. Sansano, O. Belmonte, J. Torres-Sospedra
Institute of New Imaging Technologies
Jaume I University
12071, Castellón, Spain
Email: [montoliu,esansano,belfern,jtorres]@uji.es
2018 International Conference on Indoor Positioning and Indoor Navigation (IPIN), France
"""

import numpy as np
import logging as lg
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
import warnings


class LTKNN():

    def __init__(self, data, visible_waps, rps, nn=3):
        """Init method for this class

        :param data: dataframe of all data that will be used
        :param visible_waps: list of visible_waps at init
        :param rps: list of reference location labels
        :param nn: nearest neighbors value for KNN
        :returns: self
        :rtype: LTKNN

        """

        # original and current fingerprints
        self.original_data = data
        self.current_data = data

        # waps
        self.original_waps = visible_waps
        self.current_waps = visible_waps

        # targets. remain same at all times
        self.rps = rps

        # store nearest neighbor
        self.nn = nn

        # store the SVR models for each missing WAP as they occur
        # key is wap name; value is svr model
        self.wap_svr = None

        # init the knn model
        self.knn = KNN(n_neighbors=self.nn)

        # fit knn
        self.knn.fit(self.current_data[self.original_waps].values.astype(int),
                     self.current_data[self.rps].values.flatten().astype(int))

    @property
    def missing_waps(self):
        return list(set(self.original_waps) - set(self.current_waps))

    @property
    def available_waps(self):
        return list(set(self.original_waps) - set(self.missing_waps))

    def __train_svr__(self):
        """
        impute current fp at index
        """

        # train SVR model
        parameter_candidates = [
            {'estimator__C': [10, 100],
             'estimator__gamma': [0.001, 0.0001],
             'estimator__kernel': ['rbf']},
        ]

        # grid search
        clf = GridSearchCV(estimator=MultiOutputRegressor(SVR()),
                           param_grid=parameter_candidates,
                           cv=3,
                           n_jobs=3)
        # fit the model
        clf.fit(self.original_data[self.available_waps],
                self.original_data[self.missing_waps])

        # print cross validation report in debug mode
        lg.debug(f"\tSVR CV BEST PARAMS:\n {clf.best_params_}")

        # save the best estimator
        self.wap_svr = clf.best_estimator_

    def __impute_online__(self, fp):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fp[self.missing_waps] = self.wap_svr.predict(
                fp[self.available_waps]
            )
        return fp

    def predict(self, fp):

        # there are some waps missing
        if len(self.missing_waps) != 0:
            fp = self.__impute_online__(fp)

        return self.knn.predict(fp)

    def update(self, visible_waps):
        """
        Prep the predictor for imputing rows in predict stage
        """

        # update current_visible_waps
        self.current_waps = visible_waps

        if len(self.missing_waps) != 0:
            # impute the columns of data based on some strategy
            self.__train_svr__()
