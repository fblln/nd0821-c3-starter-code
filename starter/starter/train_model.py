# Script to train machine learning model.

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import evaluate_slices, train_model, compute_model_metrics, inference

def main():
    # Load the data
    data = pd.read_csv("data/census.csv")
    data.columns = data.columns.str.strip()

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Process the training data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    # Process the test data
    X_test, y_test, _, _ = process_data(
        test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
    )

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Model Performance:\nPrecision: {precision}\nRecall: {recall}\nF1 Score: {fbeta}")

    # Evaluate model performance on slices of the data
    print("\nEvaluating model performance on data slices:")
    slice_results = evaluate_slices(
        model=model,
        encoder=encoder,
        lb=lb,
        X=test,
        categorical_features=cat_features,
        label="salary"
    )

    # Print the results
    for feature, results in slice_results.items():
        print(f"\nFeature: {feature}")
        for value, metrics in results.items():
            print(f"  Value: {value}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            print(f"    F1 Score: {metrics['fbeta']:.4f}")

    # Save the model, encoder, and label binarizer
    with open("model/model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open("model/encoder.pkl", "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)

    with open("model/lb.pkl", "wb") as lb_file:
        pickle.dump(lb, lb_file)

if __name__ == "__main__":
    main()
