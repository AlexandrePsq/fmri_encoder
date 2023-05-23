import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import os
import joblib
from omegaconf import OmegaConf

from sklearn.pipeline import Pipeline

from fmri_encoder.metrics import get_metric
from fmri_encoder.utils import check_folder, read_yaml
from fmri_encoder.loaders import get_linearmodel

from fmri_encoder.logger import console


class Encoder(object):
    def __init__(self, linearmodel, saving_folder=None, verbose=True, **model_params):
        """General class to fit linear encoding models including 'GLM' and 'Ridge', or other custom method.
        Args:
            - linearmodel: str (or custom function)
            - **model_params: dict
        """
        if verbose:
            console.log(f"Instantiating Encoder model. Saved in {saving_folder}")
        self.linearmodel = get_linearmodel(linearmodel, **model_params)
        self.is_fitted = False
        if saving_folder is not None:
            check_folder(saving_folder)
        self.saving_folder = saving_folder
        self.verbose = verbose

    def fit(self, X, y):
        """Fit the encoding model using Features X and fmri data y.
        Args:
            - X: np.Array (#samples * #features)
            - y: np.Array (#scans * #voxels)
            - gentles: list of np.Array
            - groups: list of list of int (specify if X and y are concateanted runs that need to be processed separately.)
            - nscans: list of int (number of scans in each run)
        """
        # Encoding model
        encoding_pipe = Pipeline(
            [
                # ("scaler", StandardScaler()),
                ("linearmodel", self.linearmodel),
            ]
        )

        # Fit
        if self.verbose:
            console.log(f"Fitting Encoder...")
        encoding_pipe.fit(X, y)

        # Saving pipes
        self.encoding_pipe = encoding_pipe
        self.is_fitted = True
        if self.saving_folder is not None:
            joblib.dump(
                self.encoding_pipe,
                os.path.join(self.saving_folder, "encoding_pipe.joblib"),
            )
            if self.verbose:
                console.log(
                    f'Encoder saved at {os.path.join(self.saving_folder, "encoding_pipe.joblib")}'
                )

    def predict(self, X):
        """Use the fitted encoding model to predict fmri data from features X.
        Args:
            - X: np.Array
        Returns:
            - Y_predicted: np.Array
        """
        if self.verbose:
            console.log(f"Predicting fMRI data using processed X...")
        prediction = self.encoding_pipe.predict(X)
        return prediction

    def eval(self, Y_predicted, Y_true, metric_name="r", axis=-1):
        """Compare the predicted ‘Y_predicted‘ with the ground truth ‘Y‘ using the specified ‘metric‘
        Args:
            - Y_predicted: np.Array
            - Y_true: np.Array
            - metric_name: str or sklearn buit-in function (metric used for the comparison)
        Returns:
            - evaluation: np.array
        """
        metric = get_metric(metric_name)
        if self.verbose:
            console.log(
                f"Evaluating the similarity between Y_predicted and Y_true, using metric {metric_name}..."
            )
        evaluation = metric(Y_predicted, Y_true, axis=axis)
        return evaluation

    def get_coef(self):
        """Retrieve the coefficients from the fitted linear model.
        Returns:
            - np.Array
        """
        if self.is_fitted:
            return self.encoding_pipe["linearmodel"].coef_
        else:
            if self.verbose:
                console.log(
                    f"Encoding model not fitted. You must first fit it using self.fit(X, y=None"
                )

    @classmethod
    def from_pretrained(cls, config):
        """Load pre-traind Encoder.
        Args:
            - config: OmegaConf config file
            Should contain the variables:
                - weights_path: str (path to encoding pipeline)
                - linearmodel: str
                - model_params: dict
        Returns:
            - encoder: Encoder
        """
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        elif ("yml" in config) or ("yaml" in config):
            config = OmegaConf.load(config)
        saving_folder = os.path.dirname(config.weights_path)
        linearmodel = get_linearmodel(config.linearmodel, **config.model_params)
        encoder = Encoder(linearmodel=linearmodel, saving_folder=saving_folder)
        encoder.encoding_pipe = joblib.load(config.weights_path)
        encoder.is_fitted = True
        encoder.saving_folder = saving_folder
        return encoder
