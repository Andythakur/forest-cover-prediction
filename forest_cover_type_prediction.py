# 1. Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 2. Load the Data
df = pd.read_csv('train.csv')

# 3. Preprocessing (No major cleaning is needed if the data is already numeric and one-hot encoded.)
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)),[X.columns[i] for i in indices])
plt.title('Top 10 important Features')
plt.show()
