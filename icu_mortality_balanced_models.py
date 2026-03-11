import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import json

# Load data
file_path = 'Data/ICU_MOrtality_unzipped/ICU_Patient_Monitoring_Mortality_Prediction_15000_cleaned.csv'
df = pd.read_csv(file_path)

# Preprocessing: Drop patient_id, handle categorical variables, fill missing values if any
df = df.drop('patient_id', axis=1)
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['admission_type'] = df['admission_type'].map({'Urgent': 0, 'Emergency': 1, 'Elective': 2})

# Fill missing values (if any)
df = df.fillna(df.median(numeric_only=True))

# Features and target
y = df['mortality_label']
# Drop highly correlated heart rate features and mortality_label from X
X = df.drop(['mortality_label', 'heart_rate_min', 'heart_rate_max'], axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Address class imbalance with SMOTE
over = SMOTE(random_state=42)
X_train_res, y_train_res = over.fit_resample(X_train, y_train)

# Logistic Regression with class_weight
logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train_res, y_train_res)
probs_logreg = logreg.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred_logreg = (probs_logreg > threshold).astype(int)
results = {} 
results['logistic_regression'] = {
    'accuracy': accuracy_score(y_test, y_pred_logreg),
    'confusion_matrix': confusion_matrix(y_test, y_pred_logreg).tolist(),
    'classification_report': classification_report(y_test, y_pred_logreg, output_dict=True)
}

print('--- Logistic Regression (Balanced, threshold=0.5) ---')
print('Accuracy:', results['logistic_regression']['accuracy'])
print('Confusion Matrix:\n', results['logistic_regression']['confusion_matrix'])
print('Classification Report:\n', classification_report(y_test, y_pred_logreg))

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_res, y_train_res)
probs_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = (probs_rf > threshold).astype(int)

results['random_forest'] = {
    'accuracy': accuracy_score(y_test, y_pred_rf),
    'confusion_matrix': confusion_matrix(y_test, y_pred_rf).tolist(),
    'classification_report': classification_report(y_test, y_pred_rf, output_dict=True)
}

print('\n--- Random Forest (Balanced, threshold=0.5) ---')
print('Accuracy:', results['random_forest']['accuracy'])
print('Confusion Matrix:\n', results['random_forest']['confusion_matrix'])
print('Classification Report:\n', classification_report(y_test, y_pred_rf))

# Plot and save confusion matrix for Logistic Regression
cm_logreg = confusion_matrix(y_test, y_pred_logreg)
plt.figure(figsize=(5,4))
plt.imshow(cm_logreg, cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar()
plt.xticks([0,1])
plt.yticks([0,1])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_logreg[i, j], ha='center', va='center', color='red')
plt.tight_layout()
plt.savefig('logreg_balanced_confusion_matrix.png')
plt.close()

# Plot and save confusion matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5,4))
plt.imshow(cm_rf, cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar()
plt.xticks([0,1])
plt.yticks([0,1])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm_rf[i, j], ha='center', va='center', color='red')
plt.tight_layout()
plt.savefig('rf_balanced_confusion_matrix.png')
plt.close()

# Save results to JSON
with open('balanced_model_results.json', 'w') as f:
    json.dump(results, f, indent=2)

