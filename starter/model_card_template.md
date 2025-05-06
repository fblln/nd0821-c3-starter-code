# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This repository contains a training pipeline specifically designed for binary classification tasks. It employs the Random Forest classifier from the scikit-learn library to predict whether an individual's income exceeds or falls below $50K annually.

The model was trained using carefully selected hyperparameters to maintain a compact size. Further performance enhancements could be achieved by conducting a grid search to identify optimal parameters.

## Intended Use

The model is designed to estimate an individual's income based on a variety of socio-economic features.

## Training Data

The training dataset, `census.csv`, is a widely recognized dataset, with detailed profiling available using pandas profiling [here](https://archive.ics.uci.edu/ml/datasets/census+income).

It comprises 14 features: 6 numerical and 8 categorical, with a total of 32,561 records.

## Evaluation Data

The evaluation process leverages a test dataset, which constitutes 20% of the Census data.

## Metrics

The model's performance was evaluated using the following metrics:
* Precision: 0.76
* Recall: 0.40
* F1 Score: 0.52

Additionally, performance was analyzed across different categories (refer to `model/metrics_by_slice.csv`). This provides insights into how the model performs for specific values of categorical features.

## Ethical Considerations

As the dataset contains census data exclusively from the US, the model is applicable only to individuals residing in the US. Furthermore, the model may reflect biases related to sex, race, native-country, and age due to imbalances in these categories within the dataset.

Predictions should be interpreted with caution, particularly when applied to individuals.

## Caveats and Recommendations

This model is limited to US residents. Additionally, certain variables in the dataset are imbalanced. To address potential biases, techniques such as balancing the dataset or employing up-sampling and down-sampling methods like SMOTE are recommended.