import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
file_path = 'Data/ICU_MOrtality_unzipped/ICU_Patient_Monitoring_Mortality_Prediction_15000.csv'
df = pd.read_csv(file_path)

# Show basic info
df.info()
print('\nMissing values per column:')
print(df.isnull().sum())
print('\nSummary statistics:')
print(df.describe(include='all'))

# Visualize class balance
plt.figure(figsize=(5,3))
sns.countplot(x='mortality_label', data=df)
plt.title('Mortality Label Distribution')
plt.savefig('mortality_label_distribution.png')
plt.show()

# Visualize numeric feature distributions
num_cols = df.select_dtypes(include=['float64', 'int64']).columns
num_cols = [col for col in num_cols if col not in ['mortality_label', 'patient_id']]
ax = df[num_cols].hist(figsize=(15,10), bins=30)
plt.tight_layout()
plt.savefig('numeric_feature_distributions.png')
plt.show()


# Correlation heatmap
plt.figure(figsize=(12,10))
sns.heatmap(df[num_cols].corr(), annot=False, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.savefig('feature_correlation_heatmap.png')
plt.show()


#Group statistics by mortality_label (numeric columns only)
print('\nGroup statistics by mortality_label:')
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
group_stats = df.groupby('mortality_label')[numeric_cols].agg(['mean', 'median'])
print(group_stats)


# Top features most correlated with mortality_label
corrs = df.corr(numeric_only=True)['mortality_label'].abs().sort_values(ascending=False)
print('\nTop features most correlated with mortality_label:')
print(corrs[1:11])  # Exclude self-correlation, show top 10

# Plot top 10 correlated features with mortality_label
plt.figure(figsize=(10,5))
corrs[1:11].plot(kind='bar')
plt.title('Top 10 Features Most Correlated with Mortality Label')
plt.ylabel('Absolute Correlation')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top10_corr_with_mortality_label.png')
plt.show()

# Summarize top features with largest mean difference between groups
mean_by_group = df.groupby('mortality_label')[numeric_cols].mean()
mean_diff = (mean_by_group.loc[1] - mean_by_group.loc[0]).abs().sort_values(ascending=False)
print('\nTop 10 features with largest mean difference between survivors and non-survivors:')
print(mean_diff.head(10))

# Save mean_diff to a CSV file
mean_diff.to_csv('mean_diff_by_mortality_label.csv', header=['abs_mean_difference'])

# Plot and save top 10 features with largest mean difference
plt.figure(figsize=(10,5))
mean_diff.head(10).plot(kind='bar')
plt.title('Top 10 Features with Largest Mean Difference')
plt.ylabel('Absolute Mean Difference')
plt.xlabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('top10_mean_diff_mortality_label.png')
plt.show()

# Data cleansing: drop columns with >30% missing, fill others with median
missing = df.isnull().mean()
drop_cols = missing[missing > 0.3].index.tolist()
if drop_cols:
    print(f'Dropping columns with >30% missing: {drop_cols}')
    df = df.drop(columns=drop_cols)
df = df.fillna(df.median(numeric_only=True))

# Save cleaned data
df.to_csv('Data/ICU_MOrtality_unzipped/ICU_Patient_Monitoring_Mortality_Prediction_15000_cleaned.csv', index=False)
print('Cleaned data saved.')
