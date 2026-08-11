# Iris Flower Classification — Machine Learning Model

**Project 02 — Pluto Academy AI & ML Internship**

A machine learning classification project that predicts the species of an Iris flower using sepal and petal measurements. Three classification models are trained, evaluated, and compared to identify the best-performing model.

## Dataset

**Source:** [Iris Flower Dataset](https://www.kaggle.com/datasets/uciml/iris)

The Iris dataset contains four meaningful numerical features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The target variable is `Species`, containing the Iris flower classes used by the dataset.

## Objective

Build a classification model for Iris flower species prediction, compare multiple machine learning algorithms using standard evaluation metrics, and analyze the best model using a confusion matrix.

## Data Loading & Preprocessing

The notebook performs the following steps:

- Loads `Iris.csv` using Pandas.
- Inspects the dataset shape, information, missing values, and class balance.
- Drops the `Id` column because it is only a row index and has no predictive value.
- Encodes the `Species` target using `LabelEncoder`.
- Splits the data into training and testing sets using an **80/20 split** with `random_state=42` and stratification.
- Uses `StandardScaler` for models that are sensitive to feature scale.

The Iris dataset is treated as a clean dataset with no significant missing-value handling required.

## Feature Analysis

The notebook analyzes feature relationships using:

- Correlation analysis
- Feature correlation heatmap
- Boxplots of each feature by species
- Random Forest feature importance

The analysis shows that **petal length and petal width are the most predictive features**. Their correlation with the encoded target is approximately **0.95–0.96**, and together they account for about **87% of Random Forest feature importance**.

## Models Used

Three classification algorithms are trained and compared:

1. **Logistic Regression**
2. **Random Forest**
3. **K-Nearest Neighbors (KNN)**

Logistic Regression and KNN use standardized features, while Random Forest is trained on the original feature values.

## Model Evaluation

The models are compared using:

- Accuracy
- Precision
- Recall
- F1 Score

The notebook sorts the models by F1 Score to identify the best-performing model.

## Key Results

- **Logistic Regression** was the best-performing model on the notebook's test split.
- Logistic Regression achieved **0.933 accuracy** and **0.933 F1 score**.
- KNN also achieved an **F1 score of 0.933**, with slightly lower precision.
- Random Forest achieved an **F1 score of 0.900**.
- The test set contains 30 samples.
- Logistic Regression correctly classified all 10 `setosa` samples and made only 2 total errors: one `versicolor` predicted as `virginica` and one `virginica` predicted as `versicolor`.

## Why the Models Perform Well

The Iris dataset is small, clean, and mostly linearly separable. The `setosa` class is particularly well separated from the other species.

The main overlap occurs between `versicolor` and `virginica`, especially in petal-related measurements. This explains the two classification errors made by the best model on the test split.

Because the test set contains only 30 samples, the small difference between model rankings should not be overinterpreted.

## Visualizations

The notebook includes:

- Feature correlation heatmap
- Feature-by-species boxplots
- Random Forest feature importance chart
- Model comparison bar chart
- Confusion matrix for the best-performing model

## Conclusion

The project demonstrates that all three tested models can classify Iris species effectively. Logistic Regression performed best on this particular train/test split, while KNN was very close and Random Forest scored slightly lower.

Petal length and petal width were the strongest predictive features. The confusion matrix also confirms that `setosa` is easily separated, while the main classification difficulty is distinguishing `versicolor` from `virginica`.

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Google Colab

## Repository Structure

```text
Iris-ML-Project-2/
├── README.md
├── Iris_ML_Project_2.ipynb
├── images/
└── .git/
```

## How to Run

1. Open `Iris_ML_Project_2.ipynb` in Google Colab.
2. Make sure `Iris.csv` is available in the working directory.
3. Run all cells from top to bottom.
4. Review the preprocessing output, feature analysis, model comparison, and confusion matrix.

## Notebook

The notebook contains the complete workflow from dataset loading and preprocessing through feature analysis, model training, evaluation, comparison, and final model analysis.

## Author:Prashanth LR

*Pluto Academy AI & ML Internship — Project 02*
