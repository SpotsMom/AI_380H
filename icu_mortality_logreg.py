import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import json

# Load data
file_path = 'Data/ICU_MOrtality_unzipped/ICU_Patient_Monitoring_Mortality_Prediction_15000.csv'
df = pd.read_csv(file_path)

# Preprocessing: Drop patient_id, handle categorical variables, fill missing values if any
df = df.drop('patient_id', axis=1)
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['admission_type'] = df['admission_type'].map({'Urgent': 0, 'Emergency': 1, 'Elective': 2})

# Fill missing values (if any)
df = df.fillna(df.median(numeric_only=True))

# Features and target
y = df['mortality_label']
X = df.drop('mortality_label', axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
results = {
    'accuracy': accuracy_score(y_test, y_pred),
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    'classification_report': classification_report(y_test, y_pred, output_dict=True)
}
print('Accuracy:', results['accuracy'])
print('Confusion Matrix:\n', results['confusion_matrix'])
print('Classification Report:\n', classification_report(y_test, y_pred))

# Save results to JSON
with open('logreg_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Plot and save confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.colorbar()
plt.xticks([0,1])
plt.yticks([0,1])
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.tight_layout()
plt.savefig('logreg_confusion_matrix.png')
plt.show()
