
# Sentiment Analysis Model Performance Report

## Traditional Machine Learning Models

### Model Comparison
                     accuracy  precision  recall      f1     auc  training_time
Logistic Regression     0.875     0.8958   0.875  0.8682  0.8000         4.4273
Random Forest           0.750     0.7500   0.750  0.7500  0.8667         1.8557
Ensemble                0.750     0.7500   0.750  0.7500  0.8667         0.8734
Gradient Boosting       0.750     0.8214   0.750  0.7083  0.7333         8.9234
SVM                     0.625     0.6562   0.625  0.6310  0.8000         4.7629
Naive Bayes             0.375     0.1406   0.375  0.2045  0.5333         5.8890

### Cross-Validation Results
- Best F1 Score: 0.8688
- CV Scores: ['0.8688', '0.8688', '0.8688', '0.8688']

## Deep Learning Models

### LSTM Model
- Accuracy: 0.3750
- AUC: 0.2000

### CNN Model
- Accuracy: 0.3750
- AUC: 0.4333

### Transformer Model
- Accuracy: 0.3750
- AUC: 0.4333

## Best Performing Model
Logistic Regression

## Recommendations
1. Logistic Regression shows the best balance of accuracy and training time
2. For production use, consider ensemble methods combining top models
3. Deep learning models show promise but require more data for optimal performance
