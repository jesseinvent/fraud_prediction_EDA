# Fraud Prediction

Accurately classify and predict a fraud and not fraud transaction with at least 80% accuracy by combining both supervise and unsupervised learning algorithms.

### Objective: 

Accurately classify and predict a fraud from transactions dataset with at least 80% accuracy using both supervised and unsupervised learning techniques.

Procedures and steps taken:

- Performed Basic EDA on transaction datasets.
- Derived features from datasets.
- Multicollinearity (VIF) check and Feature Importance using RandomForest.
- Supervised learning models training and evaluation.
- Neural networks training and evaluation.
- Dimensionality Reduction and Feature Extraction using PCA and tSNE.
- Used Autoencoders for Anomaly Detection only on non-fraud transactions.
- Extracted latent feature from trained encoders.
- Calculated reconstruction error.
- Combine derived latent features and reconstruction error into the main derived dataset for more accurate classification.
- Retrain all supervised learning models using final combined datasets.