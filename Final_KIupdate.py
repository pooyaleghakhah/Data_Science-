# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from imblearn.over_sampling import SMOTE

# Define output directory for saving files
script_directory = Path(__file__).parent  # Path(".")
output_folder = script_directory / "outputs"
output_folder.mkdir(exist_ok=True)  # Create directory if it does not exist

# Find and load the CSV dataset
csv_file = script_directory / "DataScience_salaries_2024.csv"
if not csv_file or not csv_file.exists():
    raise FileNotFoundError("❌ Error: The dataset file was not found!")

print(f"✅ Using dataset file: {csv_file}")
df = pd.read_csv(csv_file)
df.dropna(inplace=True)
df.columns = df.columns.str.lower().str.replace(" ", "_")
df.drop_duplicates(inplace=True)

# Convert numerical columns to the correct types
df['work_year'] = df['work_year'].astype(int)
df['salary_in_usd'] = df['salary_in_usd'].astype(float)
df['remote_ratio'] = df['remote_ratio'].astype(int)
df.drop(columns=['salary', 'salary_currency'], inplace=True, errors='ignore')

# Save dataset information
dataset_info_path = output_folder / "dataset_info.csv"
dataset_info = {
    "Columns": df.columns.tolist(),
    "Missing Values": df.isnull().sum().tolist(),
    "Data Types": df.dtypes.tolist()
}
pd.DataFrame(dataset_info).to_csv(dataset_info_path, index=False)
print(f"📄 Dataset info saved: {dataset_info_path}")

# Save processed dataset
processed_data_path = output_folder / "processed_dataset.csv"
df.to_csv(processed_data_path, index=False)
print(f"📄 Processed dataset saved: {processed_data_path}")

# Normalize salary for regression
scaler = StandardScaler()
if 'salary_in_usd' in df.columns:
    df['salary_normalized'] = scaler.fit_transform(df[['salary_in_usd']])
else:
    raise KeyError("❌ Error: 'salary_in_usd' column not found!")

normalized_data_path = output_folder / "normalized_dataset.csv"
df.to_csv(normalized_data_path, index=False)
print(f"📄 Normalized dataset saved: {normalized_data_path}")

# Classification setup
target_classification = 'experience_level'
if target_classification not in df.columns:
    raise KeyError(f"❌ Error: Column '{target_classification}' not found!")

X_class = df.drop(columns=[target_classification, 'salary_normalized'], errors='ignore').select_dtypes(include=[np.number])
Y_class = df[target_classification]
X_train_class, X_test_class, Y_train_class, Y_test_class = train_test_split(X_class, Y_class, test_size=0.2, random_state=42, stratify=Y_class)

# Handle class imbalance using SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_class, Y_train_class)

# Train Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train_resampled, Y_train_resampled)
Y_pred_dt = dt_model.predict(X_test_class)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1)
rf_model.fit(X_train_resampled, Y_train_resampled)
Y_pred_rf = rf_model.predict(X_test_class)

# Save classification reports
classification_reports_path = output_folder / "classification_reports.csv"
classification_reports = {
    "Decision Tree": classification_report(Y_test_class, Y_pred_dt, output_dict=True),
    "Random Forest": classification_report(Y_test_class, Y_pred_rf, output_dict=True)
}
pd.DataFrame(classification_reports).to_csv(classification_reports_path, index=False)
print(f"📄 Classification reports saved: {classification_reports_path}")

# Regression setup
target_regression = 'salary_normalized'
X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(df.drop(columns=[target_regression], errors='ignore').select_dtypes(include=[np.number]), df[target_regression], test_size=0.2, random_state=42, shuffle=True)

# Define regression models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001),
    "Random Forest Regression": RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
}

# Train regression models and save results
regression_results = []
for name, model in models.items():
    model.fit(X_train_reg, Y_train_reg)
    Y_pred = model.predict(X_test_reg)
    mse = mean_squared_error(Y_test_reg, Y_pred)
    r2 = r2_score(Y_test_reg, Y_pred)
    cv_scores = cross_val_score(model, X_train_reg, Y_train_reg, cv=5, scoring='r2')
    regression_results.append([name, mse, r2, np.mean(cv_scores)])

    # Save regression error distribution plots
    fig_path = output_folder / f"{name}_Error_Distribution.png"
    plt.figure(figsize=(6,5))
    sns.histplot(Y_test_reg - Y_pred, bins=30, kde=True)
    plt.title(f"{name} Prediction Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.savefig(fig_path)
    plt.close()
    print(f"📊 Figure saved: {fig_path}")

# Save regression results
regression_results_path = output_folder / "regression_results.csv"
regression_df = pd.DataFrame(regression_results, columns=["Model", "MSE", "R²", "Mean CV R²"])
regression_df.to_csv(regression_results_path, index=False)
print(f"📄 Regression results saved: {regression_results_path}")

print("✅ All files are now correctly saved in the 'outputs' folder!")
