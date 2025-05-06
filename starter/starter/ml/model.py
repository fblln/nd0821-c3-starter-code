from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Define a small Random Forest model
    model = RandomForestClassifier(
        n_estimators=10,  # Small number of trees
        max_depth=5,  # Limit the depth of each tree
        random_state=42,  # For reproducibility
    )
    model.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    predictions = model.predict(X)
    return predictions


def evaluate_slices(model, encoder, lb, X, categorical_features, label):
    """
    Compute performance metrics for slices of the data based on categorical features.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `categorical_features`.
    categorical_features : list[str]
        List containing the names of the categorical features.
    label : str
        Name of the label column in `X`.

    Returns
    -------
    metrics_by_slice : dict
        A dictionary containing precision, recall, and F1 score for each category in each categorical column.
    """
    metrics_by_slice = {}

    for column in categorical_features:
        column_metrics = {}
        for category in X[column].unique():
            # Filter data for the current category
            filtered_data = X[X[column] == category]

            # Process the filtered data
            X_processed, y, _, _ = process_data(
                X=filtered_data,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )

            # Generate predictions
            predictions = inference(model, X_processed)

            # Calculate metrics
            precision, recall, fbeta = compute_model_metrics(y, predictions)

            # Store metrics for the current category
            column_metrics[category] = {
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            }

        # Add metrics for the current column to the result
        metrics_by_slice[column] = column_metrics

    return metrics_by_slice
