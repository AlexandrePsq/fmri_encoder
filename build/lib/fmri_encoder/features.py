import os
import joblib
import logging
import numpy as np


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from nilearn.glm.first_level import compute_regressor

from fmri_encoder.utils import get_reduction_method

logging.basicConfig(filename='loggings.log', level=logging.INFO)


class FMRIPipe(BaseEstimator, TransformerMixin):
    def __init__(self, fmri_reduction_method=None, fmri_ndim=None, saving_folder='./derivatives'):
        """Set the preprocessing pipeline for fMRI data.
        Args:
            - fmri_reduction_method:str
            - fmri_ndim: int
            - saving_folder: str
        Returns:
            - features_pipe: Pipeline
        """
        self.fmri_reduction_method = fmri_reduction_method
        self.fmri_ndim = fmri_ndim
        self.fmri_pipe = None
        self.saving_folder = saving_folder
    
    def fit(self, X, y=None):
        """
        Args:
            - X: np.Array
            - y: np.Array (unused)
            - mask: np.Array
        """
        self.fmri_pipe = Pipeline(
            [
                ("selector", FeatureSelector()),  # Select non nan and non-constant values
                ("scaler", StandardScaler()),
                ("reductor", DimensionReductor(
                    method=self.fmri_reduction_method,
                    ndim=self.fmri_ndim,
                    )
                ) # reduce dimension along column axis
            ]
        )
        self.fmri_pipe.fit(X, y=y)
        joblib.dump(
            self.fmri_pipe,
            os.path.join(self.saving_folder, "fmri_pipe.joblib"),
        )

    def transform(self, X):
        """Remove the identified features learnt when calling the ‘fit‘ module.
        Args:
            - X: np.Array
            - y: np.Array (unused)
        Returns:
            - np.Array
        """
        X = self.fmri_pipe.transform(X)
        
        return X

    def fit_transform(self, X, y=None):
        """Apply ‘.fit‘ and then ‘.transform‘
        Args:
            - X: np.Array
            - y: np.Array (unused)
        Returns:
            - np.Array
        """
        self.fit(X, y=y)
        return self.transform(X)


class FeaturesPipe(BaseEstimator, TransformerMixin):
    def __init__(self, features_reduction_method=None, features_ndim=None, saving_folder='./derivatives'):
        """Set the preprocessing pipeline for features.
        Args:
            - features_reduction_method:str
            - features_ndim: int
            - saving_folder: str
        Returns:
            - features_pipe: Pipeline
        """
        self.features_reduction_method = features_reduction_method
        self.features_ndim = features_ndim
        self.dm = None
        self.features_pipe = None
        self.saving_folder = saving_folder

    def fit(self, X, y=None):
        """
        Args:
            - X: np.Array
            - y: np.Array (unused)
            - mask: np.Array
        """
        self.features_pipe = Pipeline(
            [
                ("scaler", StandardScaler()), # may be remove it form the pipe to fit it each time (ask Bertrand)
                ("reductor", DimensionReductor(
                    method=self.features_reduction_method,
                    ndim=self.features_ndim,
                    )
                ), # reduce dimension along column axis
            ]
        )
        self.features_pipe.fit(X, y=y)
        joblib.dump(
            self.features_pipe,
            os.path.join(self.saving_folder, "features_pipe.joblib"),
        )

    def transform(self, X, y=None, encoding_method='hrf', tr=2, groups=None, gentles=None, nscans=None):
        """Remove the identified features learnt when calling the ‘fit‘ module.
        Args:
            - X: np.Array
            - y: np.Array (unused)
            - encoding_method: str
            - tr: float
            - groups: list of int
            - gentles: list of np.Arrays
            - nscans: list of int
        Returns:
            - np.Array
        """
        dm = Pipeline(
            [
                ("make_design_matrix", DesignMatrixBuilder(
                    method=encoding_method,
                    tr=tr,
                    groups=groups,
                    gentles=gentles,
                    nscans=nscans,
                )),
                ("scaler2", StandardScaler()),
            ]
        )
        X = self.features_pipe.transform(X)
        X = dm.fit_transform(X, y=y)
        
        return X

    def fit_transform(self, X, y=None, encoding_method='hrf', tr=2, groups=None, gentles=None, nscans=None):
        """Apply ‘.fit‘ and then ‘.transform‘
        Args:
            - X: np.Array
            - y: np.Array (unused)
            - encoding_method: str
            - tr: float
            - groups: list of int
            - gentles: list of np.Arrays
            - nscans: list of int
        Returns:
            - np.Array
        """
        self.fit(X, y=y)
        return self.transform(X, y, encoding_method, tr, groups, gentles, nscans)


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value=np.nan):
        self.fill_value = fill_value

    def fit(self, X, y=None, mask=None):
        """Learn the features to remove. Use pre-selected features, constant features and features having nan.
        Args:
            - X: np.Array
            - y: np.Array (unused)
            - mask: np.Array
        """
        logging.info(f'Identifying unuseful features: pre-selected features, constant features and features having nan...')
        std = X.std(0)
        self.n_init_features = X.shape[1]
        if mask is None:
            self.mask = np.isnan(std) | (std == 0.0) 
        else:
            self.mask = np.isnan(std) | (std == 0.0) | mask==1
        self.valid_idx = np.where(~self.mask)[0]
        self.n_valid_features = len(self.valid_idx)
        return self

    def transform(self, X, y=None):
        """Remove the identified features learnt when calling the ‘fit‘ module.
        Args:
            - X: np.Array
            - y: np.Array (unused)
        Returns:
            - np.Array
        """
        logging.info(f'Removing unuseful features...')
        return X[:, self.valid_idx]

    def fit_transform(self, X, y=None):
        """Apply ‘.fit‘ and then ‘.transform‘
        Args:
            - X: np.Array
            - y: np.Array (unused)
        Returns:
            - np.Array
        """
        self.fit(X, y=y)
        return self.transform(X)

    def inverse_transform(self, X, y=None):
        """Reconstruct the original matrix from its reduced version by fillinf removed features with ‘self.fill_value‘.
        Args:
            - X: np.Array
            - y: np.Array (unused)
        """
        logging.info(f'Reconstructing features...')
        assert X.shape[1] == self.n_valid_features
        out = np.ones((len(X), self.n_init_features)) * self.fill_value
        out[:, self.valid_idx] = X
        return out


class DesignMatrixBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, method='hrf', tr=2, groups=None, gentles=None, nscans=None):
        """Instanciate a class to create design matrices from generated embedding representations.
        Args:
            - method: str
            - tr: float
            - groups: list of list of int (indexes in feature space)
        """
        self.method = method
        self.tr = tr
        self.groups = groups
        self.gentles = gentles
        self.nscans = nscans

    def compute_dm_hrf(self, X, nscan, gentle, oversampling=30):
        """Compute the design matrix using the HRF kernel from SPM.
        Args:
            - X: np.Array (generated embeddings, size: (#samples * #features))
            - nscan: int
            - gentle: np.Array (#samples)
            - oversampling: int
        Returns:
            - dm: np.Array
        """
        dm = np.concatenate(
                    [
                        compute_regressor(
                            exp_condition=np.vstack(
                                (
                                    gentle,
                                    np.zeros(X.shape[0]),
                                    X[:, index],
                                )
                            ),
                            hrf_model="spm",
                            frame_times=np.arange(
                                0.0,
                                nscan * self.tr - gentle[0] // self.tr,
                                self.tr,
                            ),
                            oversampling=oversampling,
                        )[
                            0
                        ]  # compute_regressor returns (signal, name)
                        for index in range(X.shape[-1])
                    ],
                    axis=1,
                )
        return dm

    def compute_dm_fir(self):
        """Compute the design matrix using the FIR method.
        """
        raise NotImplementedError()
    
    def fit(self, X=None, y=None):
        pass
    
    def transform(self, X):
        """Transform input data
        """
        if self.method =='hrf':
            output = []
            for i, group in enumerate(self.groups):
                X_ = X[group, :]
                gentle = self.gentles[i]
                if isinstance(gentle, str):
                    gentle  = np.load(gentle)
                nscan = self.nscans[i]
                output.append(self.compute_dm_hrf(X_, nscan, gentle, oversampling=30))
            X = np.concatenate(output)
        elif self.method =='fir':
            X = self.compute_dm_fir()
        return X
    
    def fit_transform(self, X, y=None):
        """Apply successively ‘.fit‘ and ‘.transform‘.
        Args:
            - X: np.Array
            - y: np.Array
        """
        self.fit(X, y)
        return self.transform(X)
            

class DimensionReductor(BaseEstimator, TransformerMixin):
    def __init__(self, method=None, ndim=None, **method_args):
        """Instanciate a dimension reduction operator.
        Args:
            - method: str
            - ndim: int
            - method_args: dict
        """
        self.method = get_reduction_method(method, ndim=ndim)
        self.ndim = ndim
    
    def fit(self, X, y=None):
        """Fit the dimension reduction operator.
        Args:
            - X: np.Array
            - y: np.Array
        """
        self.method.fit(X, y)
    
    def transform(self, X):
        """Transform X using the fitted dimension reduction operator.
        Args:
            - X: np.Array
        """
        return self.method.transform(X)

    def fit_transform(self, X, y=None):
        """Apply successively ‘.fit‘ and ‘.transform‘.
        Args:
            - X: np.Array
            - y: np.Array
        """
        self.method.fit(X, y)
        return self.method.transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform X using the fitted dimension reduction operator.
        Args:
            - X: np.Array
        """
        return self.method.inverse_transform(X)
