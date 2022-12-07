import logging

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import get_metric
from .utils import get_linearmodel
from .data import FeatureSelector, DesignMatrixBuilder, DimensionReductor

logging.basicConfig(filename='loggings.log', level=logging.INFO)


class Encoder(object):
    def __init__(self, linearmodel, reduction_method=None, fmri_ndim=None, features_ndim=None, encoding_method='hrf', tr=2., **model_params):
        """General class to fit linear encoding models including 'GLM' and 'Ridge', or other custom method.
        Args:
            - linearmodel: str (or custom function)
            - reduction_method: str:
            - fmri_ndim: int
            - features_ndim: int
            - **model_params: dict
        """
        self.linearmodel = get_linearmodel(linearmodel, **model_params)
        self.reduction_method = reduction_method
        self.fmri_ndim = fmri_ndim
        self.features_ndim = features_ndim
        self.encoding_method = encoding_method
        self.tr = tr
        self.is_fitted = False

    def fit(self, X, y, groups=None, gentles=None, nscans=None):
        """Fit the encoding model using Features X and fmri data y.
        Args:
            - X: np.Array (#samples * #features)
            - y: np.Array (#scans * #voxels)
            - groups: list of list of int (specify if X and y are concateanted runs that need to be processed separately.)
            - nscans= list of int (number of scans in each run)
        """

        # Preprocessing FMRI data: voxel selection + Scaler (+ reduction)
        fmri_pipe = Pipeline(
            [
                ("selector", FeatureSelector()),  # Select non nan and non-constant values
                ("scaler", StandardScaler()),
                ("reductor", DimensionReductor(
                    method=self.reduction_method,
                    ndim=self.fmri_ndim,
                    )
                ) # reduce dimension along column axis
            ]
        )
        # Preprocessing predictive features: voxel selection + Scaler (+ reduction)
        features_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reductor", DimensionReductor(
                    method=self.reduction_method,
                    ndim=self.features_ndim,
                    )
                ), # reduce dimension along column axis
                ("make_design_matrix", DesignMatrixBuilder(
                    method=self.encoding_method,
                    tr=self.tr,
                    groups=groups,
                    gentles=gentles,
                    nscans=nscans,
                ))
            ]
        )
        # Encoding model
        encoding_pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("linearmodel", self.linearmodel),
            ]
        )
        

        # Fit
        logging.info(f'Fitting all pipelines...')
        encoding_pipe.fit(
            features_pipe.fit_transform(X), 
            fmri_pipe.fit_transform(y)
        )

        # Saving pipes
        self.features_pipe = features_pipe
        self.fmri_pipe = fmri_pipe
        self.encoding_pipe = encoding_pipe
        self.is_fitted = True
        
    def predict(self, X):
        """Use the fitted encoding model to predict fmri data from features X.
        Args:
            - X: np.Array 
        Returns:
            - Y_predicted: np.Array
        """
        logging.warning(f'X should not be process. It should be the embeddings derived from the generator model: before building the design matrix.')
        logging.info(f'Processing features X...')
        X = self.features_pipe.transform(X)
        logging.info(f'Predicting fMRI data using processed X...')
        prediction = self.encoding_pipe.predict(X)
        return prediction
    
    def eval(self, Y_predicted, Y, metric_name='r'):
        """Compare the predicted ‘Y_predicted‘ with the ground truth ‘Y‘ using the specified ‘metric‘
        Args:
            - Y_predicted: np.Array
            - Y: np.Array
            - metric_name: str or sklearn buit-in function (metric used for the comparison)
        Returns:
            - evaluation: np.array
        """
        logging.warning(f'Y should not be process. It should be the fMRI data before voxels selection, scaling and dimension reduction.')
        logging.info(f'Processing fMRI data Y...')
        Y_true = self.fmri_pipe.transform(Y)
        metric = get_metric(metric_name)
        logging.info(f'Evaluating the match between Y_predicted and Y_true...')
        evaluation = metric(Y_predicted, Y_true)
        return evaluation

    def get_coef(self):
        """Retrieve the coefficients from the fitted linear model.
        Returns:
            - np.Array
        """
        if self.is_fitted:
            return self.encoding_pipe['linearmodel'].coef_
        else:
            logging.error(f'Encoding model not fitted. You must first fit it using self.fit(X, y=None')