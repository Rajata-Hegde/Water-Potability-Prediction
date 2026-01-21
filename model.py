import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import missingno as msno
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import joblib

data = pd.read_csv("water_potability.csv")

# Displaying basic information about the dataset
print("First 5 rows of the dataset:")
print(data.head())  # Show the first 5 rows
print("\nSummary Statistics:")
print(data.describe().T)  # Transposed summary statistics for better readability
print("\nDataset Information:")
print(data.info())  # Overview of data types and non-null counts

# Pie chart for Potability distribution
d = pd.DataFrame(data["Potability"].value_counts()).reset_index()
d.columns = ["Potability", "count"]  # Rename columns for clarity
d["Potability"] = d["Potability"].map({0: "Not Potable", 1: "Potable"})  # Map 0 and 1 to labels
fig = px.pie(
    d,
    values="count",  # Counts for pie chart
    names="Potability",  # Potability labels
    hole=0.4,  # Donut hole
    opacity=0.8,
    labels={"Potability": "Water Potability", "count": "Number of Samples"},
)
fig.update_layout(title=dict(text="Pie Chart of Potability Feature"))
fig.update_traces(textposition="outside", textinfo="percent+label")
fig.show()

# Correlation matrix with heatmap
print(data.corr())  # Display correlation matrix
sns.clustermap(data.corr(), cmap="vlag", dendrogram_ratio=(0.1, 0.2), annot=True, linewidth=0.8, figsize=(9, 10))
plt.show()
plt.figure(figsize=(15, 20))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

# List of colors for different columns
colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'brown', 'pink']

# Loop through each column in the dataset
o = 1
for i, col in enumerate(data.columns):
    plt.subplot(5, 2, o)
    sns.histplot(data=data, x=col, kde=True, color=colors[i % len(colors)], bins=20)
    plt.title(f'Distribution of {col}')
    o += 1

# Display the plots
plt.tight_layout()
plt.show()


# Load the dataset

# Set the Seaborn style
sns.set_style("whitegrid")

# Pairplot of Features by Potability
sns.pairplot(data=data, hue='Potability', palette='Set1')
plt.suptitle('Pairplot of Features by Potability', y=1.02)
plt.show()
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
# Box plots for numerical variables
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=data, x='Potability', y=col)
    plt.title(f'Box Plot of {col} by Potability')
    plt.show()


# Handle missing data
msno.matrix(data)
plt.show()
print(data.isnull().sum())  # Missing values per column
data.dropna(inplace=True)
print(data.isnull().sum())  # Verify removal of missing values

# Train/test split before fitting transformers to avoid leakage
X = data.drop(columns=['Potability'])
y = data['Potability']
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
train_idx, test_idx = next(sss.split(X, y))
X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Fit transformers on train only
scaler = MinMaxScaler()
polynom = PolynomialFeatures(degree=2)

X_train_poly = polynom.fit_transform(X_train_raw)
X_train_scaled = scaler.fit_transform(X_train_poly)

# Apply same transforms to test
X_test_poly = polynom.transform(X_test_raw)
X_test_scaled = scaler.transform(X_test_poly)

# Save transformers for inference
joblib.dump(scaler, "scaler.pkl")
joblib.dump(polynom, "polynom.pkl")

# Oversample minority class on the transformed train set only
train_df = pd.DataFrame(X_train_scaled)
train_df['Potability'] = y_train.values
class_counts = train_df['Potability'].value_counts()
minority = train_df[train_df['Potability'] == 1]
majority = train_df[train_df['Potability'] == 0]
minority_up = resample(minority, replace=True, n_samples=class_counts[0], random_state=42)
train_balanced = pd.concat([majority, minority_up]).sample(frac=1, random_state=42)  # shuffle

X_train_bal = train_balanced.drop(columns=['Potability']).values
y_train_bal = train_balanced['Potability'].values

# Create a validation split from the balanced train for model selection
val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, val_idx = next(val_split.split(X_train_bal, y_train_bal))
X_tr, X_val = X_train_bal[tr_idx], X_train_bal[val_idx]
y_tr, y_val = y_train_bal[tr_idx], y_train_bal[val_idx]

# Model fitting
model_extra = ExtraTreesClassifier(random_state=42, n_estimators=300)
model_extra.fit(X_tr, y_tr)

# Fit Random Forest with GridSearchCV
rf_classifier = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
}
grid_search_rf = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=3, verbose=1, scoring='accuracy')
grid_search_rf.fit(X_tr, y_tr)
model_random = grid_search_rf.best_estimator_

# Fit AdaBoost Classifier
model_ada = AdaBoostClassifier(random_state=42, n_estimators=200, learning_rate=1)
model_ada.fit(X_tr, y_tr)

# Fit Logistic Regression
model_log = LogisticRegression(random_state=42, max_iter=400)
model_log.fit(X_tr, y_tr)

# Fit SVM Classifier with GridSearchCV (probability enabled for metrics)
svm_classifier = SVC(probability=True)
param_grid_svm = {
    'C': [0.1, 1, 10, 200],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
grid_search_svm = GridSearchCV(estimator=svm_classifier, param_grid=param_grid_svm, cv=3, scoring='accuracy')
grid_search_svm.fit(X_tr, y_tr)
model_svm = grid_search_svm.best_estimator_

# Fit XGBoost Classifier
model_xgb = XGBClassifier(random_state=42, n_estimators=300, eval_metric='logloss')
model_xgb.fit(X_tr, y_tr)

# Evaluate on validation set for model selection
models = {
    'Extra Trees': model_extra,
    'RandomForest (GridSearch)': model_random,
    'Logistic Regression': model_log,
    'AdaBoost': model_ada,
    'SVM (GridSearch)': model_svm,
    'XGBoost': model_xgb
}

val_scores = {}
for name, model in models.items():
    preds = model.predict(X_val)
    val_scores[name] = accuracy_score(y_val, preds)

# Plot comparison using a bar graph (validation accuracy)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(val_scores.keys()), y=list(val_scores.values()), palette="viridis")
plt.title('Model Comparison - Validation Accuracy')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Choose best model by validation accuracy and refit on full balanced train
best_model_name = max(val_scores, key=val_scores.get)
best_model = models[best_model_name]
best_model.fit(X_train_bal, y_train_bal)

joblib.dump(best_model, "l.pkl")

print(f"Best Model: {best_model_name}")
print(f"Validation accuracy: {val_scores[best_model_name]:.2f}")
print("-------------------------------------------------")

# Final evaluation on held-out test set
y_test_pred = best_model.predict(X_test_scaled)
y_test_proba = best_model.predict_proba(X_test_scaled)[:, 1]

accuracy_test = accuracy_score(y_test, y_test_pred)
precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
pr_auc = auc(recall, precision)
roc_auc = auc(*roc_curve(y_test, y_test_proba)[:2])
confusion = confusion_matrix(y_test, y_test_pred)
classification_rep = classification_report(y_test, y_test_pred)

print(f"Test accuracy: {accuracy_test:.2f}")
print(f"Test ROC AUC: {roc_auc:.2f}")
print(f"Test PR AUC: {pr_auc:.2f}")
print("-------------------------------------------------")
print("Confusion Matrix:\n", confusion)
print("-------------------------------------------------")
print("Classification Report:\n", classification_rep)

# Heatmap for confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=["Not Potable", "Potable"], yticklabels=["Not Potable", "Potable"])
plt.title(f"Confusion Matrix - {best_model_name} Test Evaluation")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC and Precision-Recall curve visualizations
fpr, tpr, _ = roc_curve(y_test, y_test_proba)

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")

# Plot Precision-Recall curve
plt.subplot(1, 3, 2)
plt.step(recall, precision, color='b', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')

# Plot F1-score curve
plt.subplot(1, 3, 3)
f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision, recall)]
plt.plot(recall, f1_scores, color='g', lw=2, label='F1-score')
plt.xlabel('Recall')
plt.ylabel('F1-score')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('F1-score Curve')
plt.legend(loc="lower right")

plt.tight_layout()
plt.show()
