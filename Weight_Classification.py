import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data_path = '/kaggle/input/raw-dataset/weight_dataset.xlsx'
data = pd.read_excel(data_path)
print("Database loaded successfully")

print(data.head())

print(data.info())

weight_stats = data['Weight'].describe() #analyzing the range of weights in the dataset
print("Weight Statistics:\n", weight_stats)

bins = [1 + i for i in range(11)]  # creates bins from 1 to 10
labels = list(range(1, 11))            # Labels for each class (1, 2, ..., 10)
data['Weight_Class'] = pd.cut(data['Weight'], bins=bins, labels=labels, right=False)

X = data.drop(columns=['Weight', 'Weight_Class', 'LABEL', 'Photo #'])  # Drop unnecessary columns
y = data['Weight_Class'] # Target variable is the weight class

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.dtypes)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Confusion Matrix")
plt.show()

