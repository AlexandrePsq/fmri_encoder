import numpy as np

from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.base import BaseEstimator, TransformerMixin

from joblib import Parallel, delayed

from fmri_encoder.logger import rich_progress_joblib, get_progress, console
from fmri_encoder.utils import check_folder


class CustomRidge(BaseEstimator, TransformerMixin):
    def __init__(self, alpha_min, alpha_max, nb_alphas, nscans):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.nb_alphas = nb_alphas
        if (nscans is None) or (len(nscans) == 1):
            self.nscans = [
                nscans[0] // 3,
                nscans[0] // 3,
                nscans[0] - 2 * nscans[0] // 3,
            ]
            console.log(f"Using by default 3 CV splits.")
        else:
            self.nscans = nscans
        self.model = Ridge()

    def fit(self, X, y):
        """Fit the model for a given set of runs.
        Arguments:
            - X: np.array
            - Y: np.array
        """
        start = 0
        self.X_train = X
        self.Y_train = y
        alpha_list = np.logspace(self.alpha_min, self.alpha_max, self.nb_alphas)

        output = []
        with get_progress(transient=True) as progress:
            task0 = progress.add_task(
                f"Cross validating the alphas", total=len(self.nscans)
            )
            for p, position in enumerate(self.nscans):
                stop = start + position
                X_train = np.vstack(
                    [X[:start, :], X[stop:, :]],
                )
                Y_train = np.vstack(
                    [y[:start, :], y[stop:, :]],
                )
                X_test = X[start:stop, :]
                Y_test = y[start:stop, :]
                start = stop
                result = {
                    "R2": [],
                    "Pearson_coeff": [],
                    "alpha": alpha_list if ("Ridge" in str(self.model)) else [1],
                }
                task1 = progress.add_task(
                    f"Fitting Encoder for each alpha", total=len(alpha_list)
                )
                for alpha in alpha_list:
                    self.model_fit(X_train, Y_train, alpha)
                    predictions = self.model.predict(X_test)
                    result["R2"].append(self.get_R2_coeff(predictions, Y_test))
                    result["Pearson_coeff"].append(
                        self.get_Pearson_coeff(predictions, Y_test, progress=progress)
                    )
                    progress.update(task1, advance=1)
                result["R2"] = np.stack(result["R2"], axis=0)
                result["Pearson_coeff"] = np.stack(result["Pearson_coeff"], axis=0)

                output.append(result)
                progress.update(task0, advance=1)

        output = np.stack(output, axis=0)
        data = np.stack([r["R2"] for r in output], axis=0)
        self.optimize_alpha(data, alpha_list)

    def model_fit(self, X_train, Y_train, alpha):
        """Fit a Ridge model by first by setting the alpha.
        Args:
            - X_train: np.Array
            - Y_train: np.Array
            - alpha: float
        """
        if "Ridge" in str(self.model):
            self.model.set_params(alpha=alpha)
        self.model.fit(X_train, Y_train)

    def optimize_alpha(self, data, hyperparameter):
        """Optimize the hyperparameter of a model given a
        list of measures.
        Arguments:
            - data: np.array (3D)
            - hyperparameter: np.array (1D)
        Returns:
            - voxel2alpha: list (of int)
            - alpha2voxel: dict (of list)
        """
        console.log(f"Identifying the best alpha for each voxel...")
        best_alphas_indexes = np.argmax(np.mean(data, axis=0), axis=0)
        voxel2alpha = np.array([hyperparameter[i] for i in best_alphas_indexes])
        alpha2voxel = {key: [] for key in hyperparameter}
        for index in range(len(voxel2alpha)):
            alpha2voxel[voxel2alpha[index]].append(index)
        self.voxel2alpha = voxel2alpha
        self.alpha2voxel = alpha2voxel

    def predict(self, X, X_train=None, Y_train=None):
        """Fit a model for each voxel given the parameter optimizing a measure.
        Arguments:
            - X_train: np.array
            - Y_train: np.array
            - X: np.array
            - Y_test: np.array
        The extra dimension of the last 3 arguments results from the ’aggregate_cv’
        method that was applied to the output of ’grid_search’, concatenating cv
        results over a new dimension placed at the index 0.
        Returns:
            - predictions: np.array
        """
        if (X_train is None) or (Y_train is None):
            X_train = self.X_train
            Y_train = self.Y_train
        predictions = np.zeros((X.shape[0], Y_train.shape[1]))

        with get_progress(transient=True) as progress:
            task = progress.add_task(
                f"Making predictions", total=len(self.alpha2voxel.items())
            )
            for alpha_, voxels in self.alpha2voxel.items():
                if voxels:
                    y_train_ = Y_train[:, voxels]
                    self.model_fit(X_train, y_train_, alpha_)
                    predictions[:, voxels] = self.model.predict(X)
                progress.update(task, advance=1)

        return predictions

    def get_R2_coeff(self, predictions, Y_test):
        """Compute the R2 score for each voxel (=list).
        Arguments:
            - predictions: np.array
            - Y_test: np.array
        """
        r2 = r2_score(Y_test, predictions, multioutput="raw_values")
        return r2

    def get_Pearson_coeff(self, predictions, Y_test, progress=None):
        """Compute the Pearson correlation coefficients
        score for each voxel (=list).
        Arguments:
            - predictions: np.array
            - Y_test: np.array
        """

        def get_corr(a, b):
            res = pearsonr(a, b)[0]
            return res

        if progress is None:
            with rich_progress_joblib(
                "Computing Pearson coefficients",
                total=Y_test.shape[1],
                verbose=True,
            ):
                pearson_corr = np.array(
                    Parallel(n_jobs=-2)(
                        delayed(get_corr)(Y_test[:, i], predictions[:, i])
                        for i in range(Y_test.shape[1])
                    )
                )
        else:
            task2 = progress.add_task("Computing Pearson coefficients", total=1)
            pearson_corr = np.array(
                Parallel(n_jobs=-2)(
                    delayed(get_corr)(Y_test[:, i], predictions[:, i])
                    for i in range(Y_test.shape[1])
                )
            )
            progress.update(task2, advance=1)

        return pearson_corr
