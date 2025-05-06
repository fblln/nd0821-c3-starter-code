import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from typing import Optional, Tuple


def process_data(
    X: pd.DataFrame,
    categorical_features: Optional[list[str]] = None,
    label: Optional[str] = None,
    training: bool = True,
    encoder: Optional[OneHotEncoder] = None,
    lb: Optional[LabelBinarizer] = None,
) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder, LabelBinarizer]:
    """
    Processes a DataFrame for ML:
      - one‑hot encodes categorical_features
      - binarizes label column

    Returns:
    --------
    X_out : np.ndarray
        [n_samples, n_continuous + n_encoded_categorical]
    y_out : np.ndarray
        [n_samples,] (empty if label is None)
    encoder : OneHotEncoder
    lb : LabelBinarizer
    """

    # avoid mutable defaults
    if categorical_features is None:
        categorical_features = []

    # --- 1) extract and drop the label column ---
    if label is not None:
        y_series = X[label]
        X = X.drop(columns=[label])
    else:
        y_series = pd.Series([], dtype=int)

    # --- 2) split out categorical vs continuous and convert to numpy ---
    X_cat = X[categorical_features].to_numpy()
    X_cont = X.drop(columns=categorical_features).to_numpy()

    # --- 3) fit or transform encoder & label binarizer ---
    if training:
        encoder = OneHotEncoder(
            sparse_output=False,  # new in sklearn ≥ 1.2; replaces `sparse=False`
            handle_unknown="ignore",
        )
        lb = LabelBinarizer()

        X_cat = encoder.fit_transform(X_cat)
        y_arr = lb.fit_transform(y_series)

        # if binary, fit_transform returns shape (n_samples, 1), so flatten:
        if y_arr.ndim == 2 and y_arr.shape[1] == 1:
            y_arr = y_arr.ravel()
    else:
        if encoder is None or lb is None:
            raise ValueError("encoder and lb must be provided for inference")

        X_cat = encoder.transform(X_cat)

        if label is not None:
            y_arr = lb.transform(y_series)
            if y_arr.ndim == 2 and y_arr.shape[1] == 1:
                y_arr = y_arr.ravel()
        else:
            y_arr = np.array([])

    # --- 4) concatenate back together ---
    X_out = np.hstack([X_cont, X_cat])

    return X_out, y_arr, encoder, lb
