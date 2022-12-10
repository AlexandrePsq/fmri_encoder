import numpy as np
import logging

from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin



class CustomRidge(BaseEstimator, TransformerMixin):
    def __init__(self, alpha_min, alpha_max, nb_alphas, nscans):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.nb_alphas = nb_alphas
        self.nscans = nscans
        self.model = Ridge()

    def fit(self, X, y):
        """ Fit the model for a given set of runs.
        Arguments:
            - X_train: list (of np.array)
            - Y_train: list (of np.array)
        """
        start = 0
        alpha_list=np.logspace(self.alpha_min, self.alpha_max, self.nb_alphas)

        output = []

        for p, position in enumerate(self.nscans):
            stop = start + position
            X_train = np.vstack([X[:start, :], X[stop:, :]], )
            Y_train = np.vstack([y[:start, :], y[stop:, :]], )
            X_test = X[start:stop, :]
            Y_test = y[start:stop, :]
            print(X.shape, y.shape, X_train.shape, Y_train.shape, X_test.shape, Y_test.shape, position)
            start = stop
            result = {'R2': [],
                        'Pearson_coeff': [],
                        'alpha': alpha_list if ('Ridge' in str(self.model)) else [1]
                        }

            for alpha in alpha_list:
                self.model_fit(X_train, Y_train, alpha)
                predictions = self.model.predict(X_test)
                result['R2'].append(self.get_R2_coeff(predictions, Y_test))
                result['Pearson_coeff'].append(self.get_Pearson_coeff(predictions, Y_test))
            result['R2'] = np.stack(result['R2'], axis=0)
            result['Pearson_coeff'] = np.stack(result['Pearson_coeff'], axis=0)

            output.append(result)

        output = np.stack(output, axis=0)
        data = np.stack([r['R2'] for r in output], axis=0)
        
        voxel2alpha, alpha2voxel = self.optimize_alpha(data, alpha_list)
        self.voxel2alpha = voxel2alpha
        self.alpha2voxel = alpha2voxel

    def model_fit(self, X_train, Y_train, alpha):
        """Fit a Ridge model by first by setting the alpha.
        Args:
            - X_train: np.Array
            - Y_train: np.Array
            - alpha: float
        """
        if 'Ridge' in str(self.model):
            self.model.set_params(alpha=alpha)
        self.model.fit(X_train,Y_train)
    
    def optimize_alpha(self, data, hyperparameter):
        """ Optimize the hyperparameter of a model given a
        list of measures.
        Arguments:
            - data: np.array (3D)
            - hyperparameter: np.array (1D)
        Returns:
            - voxel2alpha: list (of int)
            - alpha2voxel: dict (of list)
        """
        best_alphas_indexes = np.argmax(np.mean(data, axis=0), axis=0)
        voxel2alpha = np.array([hyperparameter[i] for i in best_alphas_indexes])
        alpha2voxel = {key:[] for key in hyperparameter}
        for index in range(len(voxel2alpha)):
            alpha2voxel[voxel2alpha[index]].append(index)
        return voxel2alpha, alpha2voxel

    def predict(self, X_train, X_test, Y_train):
        """ Fit a model for each voxel given the parameter optimizing a measure.
        Arguments:
            - X_train: list (of np.array)
            - Y_train: list (of np.array)
            - X_test: list (of np.array)
            - Y_test: list (of np.array)
        The extra dimension of the last 3 arguments results from the ’aggregate_cv’
        method that was applied to the output of ’grid_search’, concatenating cv
        results over a new dimension placed at the index 0.
        Returns:
            - result: dict
        """
        predictions = np.zeros((X_test.shape[0], Y_train.shape[1]))
        coefs = np.zeros((Y_train.shape[1], X_test.shape[1]))
        intercepts = np.zeros((Y_train.shape[1]))
        diag_matrix = np.array([])

        for alpha_, voxels in self.alpha2voxel.items():
            if voxels:
                y_train_ = Y_train[:, voxels]
                self.model_fit(X_train, y_train_, alpha_)
                predictions[:, voxels] = self.model.predict(X_test)
                coefs[voxels, :] = self.model.coef_
                intercepts[voxels] = self.model.intercept_
        result = {
                    'predictions': predictions,
                    'coef_': coefs,
                    'intercept_': intercepts,
                    'diag_matrix': diag_matrix,
                    }
        return result

    def get_R2_coeff(self, predictions, Y_test):
        """ Compute the R2 score for each voxel (=list).
        Arguments:
            - predictions: np.array
            - Y_test: np.array
        """
        r2 = r2_score(Y_test, predictions, multioutput='raw_values')
        return r2

    def get_Pearson_coeff(self, predictions, Y_test):
        """ Compute the Pearson correlation coefficients
        score for each voxel (=list).
        Arguments:
            - predictions: np.array
            - Y_test: np.array
        """
        pearson_corr = np.array([pearsonr(Y_test[:,i], predictions[:,i])[0] for i in range(Y_test.shape[1])])
        return pearson_corr