import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import get_metric
from .utils import get_linearmodel

logging.basicConfig(filename='loggings.log', level=logging.INFO)


class Encoder(object):
    def __init__(
        self, 
        linearmodel, 
        **model_params
        ):
        """General class to fit linear encoding models including 'GLM' and 'Ridge', or other custom method.
        Args:
            - linearmodel: str (or custom function)
            - **model_params: dict
        """
        self.linearmodel = get_linearmodel(linearmodel, **model_params)
        self.is_fitted = False

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
                #("scaler", StandardScaler()),
                ("linearmodel", self.linearmodel),
            ]
        )

        # Fit
        logging.info(f'Fitting all pipelines...')
        encoding_pipe.fit(X,y)

        # Saving pipes
        self.encoding_pipe = encoding_pipe
        self.is_fitted = True
        
    def predict(self, X):
        """Use the fitted encoding model to predict fmri data from features X.
        Args:
            - X: np.Array 
        Returns:
            - Y_predicted: np.Array
        """
        logging.info(f'Predicting fMRI data using processed X...')
        prediction = self.encoding_pipe.predict(X)
        return prediction
    
    def eval(self, Y_predicted, Y_true, metric_name='r'):
        """Compare the predicted ‘Y_predicted‘ with the ground truth ‘Y‘ using the specified ‘metric‘
        Args:
            - Y_predicted: np.Array
            - Y_true: np.Array
            - metric_name: str or sklearn buit-in function (metric used for the comparison)
        Returns:
            - evaluation: np.array
        """
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
