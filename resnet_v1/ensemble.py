import pandas as pd
import numpy as np
import random
import csv
# Load predictions from different models

predictions_model1 = pd.read_csv('./performance_top_five/7764.csv')  # Assuming CSV format
predictions_model2 = pd.read_csv('./performance_top_five/7686.csv')
predictions_model3 = pd.read_csv('./performance_top_five/7646.csv')
predictions_model4 = pd.read_csv('./performance_top_five/7658.csv')
predictions_model5 = pd.read_csv('./performance_top_five/7723.csv')


# Perform majority voting
ensemble_predictions = []
tie_cases = []

for row_idx in range(len(predictions_model1)):
    labels = np.array([predictions_model1.loc[row_idx, 'category'],
              predictions_model2.loc[row_idx, 'category'],
              predictions_model3.loc[row_idx, 'category'],
              predictions_model4.loc[row_idx, 'category'],
              predictions_model5.loc[row_idx, 'category']])
    # print(labels)
    label_counts = np.bincount(labels,minlength=4)
    # print(label_counts)
    # print(label_counts.shape)
    max_count = np.max(label_counts)
    
    if np.sum(label_counts == max_count) == 1:
        # Majority vote
        ensemble_label = np.argmax(label_counts)
    else:
        # Tie case
        tie_cases.append(row_idx)
        ensemble_label = labels[0]
    
    ensemble_predictions.append(ensemble_label)

# Save ensemble predictions
ensemble_results = pd.DataFrame({'id': predictions_model1['id'], 
                                 'category': ensemble_predictions})
ensemble_results.to_csv('ensemble_results_resnet_aug_top5_3.csv',index=False)

print("Tie cases occurred at row numbers:", tie_cases)
