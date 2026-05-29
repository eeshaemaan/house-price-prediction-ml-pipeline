# house-price-prediction-ml-pipeline

A complete Regression Machine Learning Project built on the Ames Housing Dataset from Kaggle — designed to predict house prices with high accuracy using advanced feature engineering and ensemble learning.

This project follows a full end-to-end ML pipeline: from raw CSV data to a production-ready, serialized model ready for deployment.

Key Features:

- Complete data cleaning & preprocessing (81 columns, messy missing values)
- Strategic missing value imputation (median for numerical, mode / 'None' for categorical)
- Engineered 26 custom features (polynomial, interaction, structural, binary flags, categorical bins)
- IQR-based outlier detection & capping on 9 key numerical features
- Hyperparameter tuning using RandomizedSearchCV with 3-fold cross-validation

Models Trained:

- Linear Regression, Ridge, Lasso, ElasticNet, Decision Tree
- Random Forest, XGBoost, LightGBM

Ensemble Methods:

- Voting Regressor with optimized weights (Ridge=3, XGBoost=1, LightGBM=1, RF=1)
- Stacking Regressor with Ridge as meta-learner

Results:

Best Model: Voting Ensemble → 91.6% R² with log-RMSE of 0.125
Explains over 91% of variance in house prices

Tech Stack:
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn (LinearRegression, Ridge, Lasso, ElasticNet, RandomForestRegressor, VotingRegressor, StackingRegressor, ColumnTransformer, SimpleImputer, StandardScaler, OneHotEncoder, RandomizedSearchCV), XGBoost, LightGBM, Joblib

Key Learning:
This project strengthened my understanding of end-to-end ML pipelines — from data cleaning and feature engineering to model tuning, ensembling, and production serialization. The insight that Ridge regression (a linear model) outperformed complex tree-based models in the ensemble was particularly valuable, showing that simple, interpretable patterns (quality + area) are the strongest predictors in real estate data.

#DataScience #MachineLearning #Regression #EnsembleLearning #Python #ScikitLearn #XGBoost #LightGBM #AmesHousing
