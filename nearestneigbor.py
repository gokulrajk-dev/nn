import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = pd.read_csv("NEAREST NEIGHBORS/Iris.csv")
X = iris['SepalLengthCm'].values.reshape(-1, 1)
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
k=3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'KNN Accuracy: {accuracy_knn:.2f}')
print('\nKNN Classification Report:')
print(classification_report(y_test, y_pred_knn))
print('\nKNN Confusion Matrix:')
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(cm_knn)
