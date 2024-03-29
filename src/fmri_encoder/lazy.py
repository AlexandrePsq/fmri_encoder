import numpy as np
from nilearn.image import clean_img
from fmri_encoder.encoder import Encoder
from fmri_encoder.data import fetch_masker
from fmri_encoder.loaders import get_groups
from sklearn.model_selection import LeavePOut
from fmri_encoder.logger import console
from fmri_encoder.features import FMRIPipe, FeaturesPipe


def default_encoder(X_train, Y_train, X_test=None, Y_test=None):
    """Run an encoder with default parameters.
    Args:
        - X_train: np.Arrays
        - Y_train: np.Arrays
        - X_test: np.Arrays
        - Y_test: np.Arrays
    """
    # Instantiating the encoding model
    linearmodel = "ridgecv"
    encoding_params = {
        "alpha": 0.001,
        "alpha_per_target": True,
    }
    encoder = Encoder(linearmodel=linearmodel, saving_folder=None, **encoding_params)
    encoder.fit(X_train, Y_train)

    # Testing on test set
    if (X_test is not None) and (Y_test is not None):
        predictions = encoder.predict(X_test)
        score = encoder.eval(predictions, Y_test, axis=0)
    else:
        predictions = None
        score = None
    return {"encoder": encoder, "score": score, "predictions": predictions}


def default_cv_encoder(X, Y, return_preds=False):
    """
    Run a cross-validated encoder with default parameters.
    Args:
        - X: list of np.Arrays
        - Y: list of np.Arrays
        - return_preds: bool
    """
    # Instantiating the encoding model
    assert len(X) == len(Y)
    out_per_fold = 1
    logo = LeavePOut(out_per_fold)
    scores = []
    predictions = []

    # Loop
    for train, test in logo.split(X):
        Y_train = np.vstack([Y[i] for i in train])
        X_train = np.vstack([X[i] for i in train])
        Y_test = np.vstack([Y[i] for i in test])
        X_test = np.vstack([X[i] for i in test])
        output = default_encoder(X_train, Y_train, X_test, Y_test)
        scores.append(output["score"])
        if return_preds:
            predictions.append(output["predictions"])
    cv_score = np.mean(np.stack(scores, axis=0), axis=0)

    return {
        "scores": scores,
        "cv_score": cv_score,
        "predictions": predictions,
    }


def default_processing(X, offsets, tr, Y=None, nscans=None, masker_path="masker"):
    """
    Run a cross-validated encoder with default parameters.
    Args:
        - X: list of np.Arrays
        - Y: list of np.Arrays
        - tr: float
        - offsets: list of np.Arrays
        - masker_path: str (path without the extension)
        - nscans: list of int
    """
    # Instantiating the encoding model
    encoding_method = "hrf"

    assert len(offsets) == len(X)
    if Y is not None:
        assert len(Y) == len(X)
    else:
        masker = None
        assert nscans is not None
        assert len(nscans) == len(X)

    if Y is not None:
        # Instantiating the fMRI data processing pipeline
        fmri_pipe = FMRIPipe(fmri_reduction_method=None, fmri_ndim=None)
        # Fetch or create a masker object that retrieve the voxels of interest in the brain
        masker = fetch_masker(masker_path, Y, **{"detrend": True, "standardize": True})

        # Preprocess fmri data with the masker
        Y = [clean_img(f, ensure_finite=True) for f in Y]
        Y = [masker.transform(f) for f in Y]
        nscans = [f.shape[0] for f in Y]  # Number of scans per session
        Y = [fmri_pipe.fit_transform(y) for y in Y]

    # Instantiating the features processing pipeline
    features_reduction_method = (
        None  # you can reduce the dimension if you want: 'pca', ...
    )
    features_ndim = None  # 100, ...
    features_pipe = FeaturesPipe(
        features_reduction_method=features_reduction_method, features_ndim=features_ndim
    )

    # Preprocess features
    X = [
        features_pipe.fit_transform(
            x,
            encoding_method=encoding_method,
            tr=tr,
            groups=get_groups([offset_x]),
            gentles=[offset_x],
            nscans=[nscan_x],
        )
        for (x, offset_x, nscan_x) in zip(X, offsets, nscans)
    ]
    return {"X": X, "Y": Y, "masker": masker, "nscans": nscans}


def default_process_and_cv_encode(
    X, Y, offsets, tr, return_preds=False, masker_path="masker"
):
    """
    Run a cross-validated encoder with default parameters.
    Args:
        - X: list of np.Arrays
        - Y: list of np.Arrays
        - tr: float
        - offsets: list of np.Arrays
        - return_preds: bool
    """
    processed_data = default_processing(X, offsets, tr, Y, masker_path=masker_path)
    X = processed_data["X"]
    Y = processed_data["Y"]

    output = default_cv_encoder(X, Y, return_preds=return_preds)

    return output


def default_process_multipleX_and_cv_encode(
    Xs, Y, offsets, tr, return_preds=False, masker_path="masker"
):
    """
    Preprocess multiple features and brain data and then run a
    cross-validated encoder with default parameters.
    Args:
        - X: list of list of np.Arrays
        - Y: list of np.Arrays
        - tr: float
        - offsets: list of list of np.Arrays
        - return_preds: bool
    """
    processed_data = default_processing(
        Xs[0], offsets[0], tr=tr, Y=Y, masker_path=masker_path
    )
    X = processed_data["X"]
    Y = processed_data["Y"]
    masker = processed_data["masker"]
    nscans = processed_data["nscans"]
    X_shapes = [[X_i.shape for X_i in X]]

    for X_i, offset_i in zip(Xs[1:], offsets[1:]):
        processed_data = default_processing(
            X_i, offset_i, tr=tr, Y=None, nscans=nscans, masker_path=masker_path
        )
        X = [np.hstack([X[j], processed_data["X"][j]]) for j in range(len(X))]
        X_shapes.append([X_j.shape for X_j in processed_data["X"]])

    X_shapes = list(zip(*X_shapes))
    logs = "\n".join(
        [
            f" \
    Run {i+1}: \
        * X.shape = {X_i}\
        * Y.shape = {Y_i.shape}"
            for i, (X_i, Y_i) in enumerate(zip(X_shapes, Y))
        ]
    )
    console.log(logs)

    output = default_cv_encoder(X, Y, return_preds=return_preds)
    output["masker"] = masker

    return output
