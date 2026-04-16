# Final Report: Prostate Cancer Diagnosis Prediction

## Introduction
This project focuses on predicting and classifying prostate cancer diagnoses using machine learning algorithms. By analyzing various physiological and structural features of cell nuclei, the goal is to build a robust model capable of differentiating between benign and malignant cases. The project pipeline involves extensive exploratory data analysis, data preprocessing (including scaling, transforming, and balancing), and feature importance evaluation to ensure high model accuracy and reliability.
## Data
The data used in this project is sourced from the Prostate_Cancer.csv dataset. It contains multiple numerical features representing characteristics of cell nuclei, such as radius, area, perimeter, compactness, and fractal_dimension. The target variable is the diagnostic result. The dataset required careful handling due to varying feature scales, skewed distributions, and a slight class imbalance in the target variable.

## Questions & Answers

### Q1: How do we handle the significantly varying scales of different features to prevent model bias?
Solved by: [Insert Name Here] Answer: After seeing the data, we got to know that the mean of the various features are varying a lot. For instance, the mean of the radius is around 16.85 and the mean of the area is around 702.88. If fed directly into a model, this would lead to overfitting or one feature dominating the distance calculations. That is the reason using StandardScaler was very important for standardizing the features so that after training a model, there should be no overfitting or feature dominance.[Insert image.png here - Feature Means/Scales]Code Snippet:from sklearn.preprocessing import StandardScaler

#### StandardScaler is added as a final step in our pipeline to standardize all features
'''scaler = StandardScaler()'''

### Q2: How should we treat features with highly skewed distributions to normalize our data?
Solved by: [Insert Name Here]Answer: We will be applying the Yeo-Johnson transformation on area, compactness, perimeter, and fractal_dimension. This was concluded by:Observing the histograms — these features showed a clear right skew with a long tail.Comparing the 75th percentile and maximum values — a large gap between the 75th percentile and the max indicates extreme outliers pulling the distribution to the right, confirming significant skewness.[Insert image.png here - Histograms showing right skew]

'''
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
'''

# Define skewed columns
skewed_cols = ['area', 'compactness', 'fractal_dimension', 'perimeter']

# Remaining columns
normal_cols = [col for col in X.columns if col not in skewed_cols]

# ColumnTransformer — YeoJohnson on skewed, passthrough rest
col_transformer = ColumnTransformer([
    ('yeo', PowerTransformer(method='yeo-johnson'), skewed_cols),
    ('pass', 'passthrough', normal_cols)
])

### Q3: Which features are most indicative and dependent for predicting the diagnostic result?
Solved by: [Insert Name Here]Answer: From the correlation matrix, we can see that the diagnostic result has a good correlation with the features compactness, area, and perimeter. We verified this relationship and got the same result regarding feature dependence by using the Random Forest classifier. When extracting the feature importance in descending order, we found the top three features to be dependent are perimeter, area, and compactness.[Insert image.png here - Correlation Matrix / Feature Importance Plot]Code Snippet:from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Assuming rf_model is a trained RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Getting feature importances
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(3)
Q4: How do we address the class imbalance in our target variable within the training data?Solved by: [Insert Name Here]Answer: Since we saw before that there was an imbalance in the data (specifically within the train data) where the 1s and 0s were 46 and 34 respectively. We did not want our model to be biased towards the majority class, thus we applied SMOTE (Synthetic Minority Over-sampling Technique) to remove that imbalance.Code Snippet:from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Pipeline steps integrating SMOTE, Transformations, and Scaling
pipeline_steps_v2 = [
    ('smote', SMOTE(random_state=42)),
    ('col_transform', col_transformer),   # YeoJohnson on skewed cols only
    ('scaler', StandardScaler())          # StandardScaler over all after
]


ReferencesMcKinney, W. (2010). Data Structures for Statistical Computing in Python (pandas).Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research.Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning.Yeo, I. K., & Johnson, R. A. (2000). A new family of power transformations to improve normality or symmetry. Biometrika.