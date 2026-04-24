# house-price-prediction-ml-pipeline
Cleaned Ames Housing dataset (81 features), engineered 26 custom features (including Area_Quality), capped outliers via IQR, and trained XGBoost, LightGBM, and Random Forest. Final Voting Ensemble achieved log-RMSE of 0.125, explaining 91% of price variance. Model + preprocessor serialized with joblib for production use.
