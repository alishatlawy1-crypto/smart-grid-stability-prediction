# ⚡ Smart Grid Stability Prediction System

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Machine Learning](https://img.shields.io/badge/Algorithm-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 1. Project Overview
In modern Mechatronics and Power Engineering, the stability of "Smart Grids" is critical. As we add more decentralized power sources (like solar/wind), the grid becomes harder to control. This project uses **Machine Learning** to predict grid instability in real-time based on the reaction speed of power nodes. By identifying unstable conditions early, control systems can prevent blackouts.

## 2. Problem Statement
* **Objective:** Classify the stability of a 4-node star network smart grid.
* **Input Data:** Reaction time (`tau`), Power output (`p`), and Elasticity (`g`).
* **Target Variable:** `stable` vs `unstable`.
* **Success Metric:** >90% Prediction Accuracy.

## 3. The Dataset
* **Source:** UCI Machine Learning Repository (Electrical Grid Stability Simulated Data).
* **Records:** 10,000 simulations.
* **Features:** 12 inputs (reaction times, power balance, damping coefficients).

## 4. Methodology
The system is built using *Python* and the *Scikit-Learn* library.
1.  *Preprocessing:* Converted text labels ('stable'/'unstable') into binary format (0/1).
2.  *Model:* Used a *Random Forest Classifier* with 100 decision trees.
3.  *Validation:* Tested on a 20% hold-out set (2,000 samples) to ensure reliability.

## 5. Results
The model achieved high accuracy in distinguishing between stable and unstable grid states.
* *Accuracy:* ~99.9%
* *Confusion Matrix:* Shows minimal False Negatives, ensuring that dangerous "Unstable" states are almost never missed.

<img width="1918" height="847" alt="git hub 2" src="https://github.com/user-attachments/assets/eeab2031-1577-4fdf-9606-98afb4c65461" />


## 6. How to Run
1.  Clone the repository.
2.  Install requirements: pip install -r requirements.txt
3.  Run the script: python main.py

## 7. Implementation (Python Code)
Below is the complete source code used to train and validate the predictive model.

```python
# Smart Grid Stability Prediction
# Algorithm: Random Forest Classifier

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# --- 1. LOAD DATA ---
# Ensure 'Data_for_UCI_named.csv' is uploaded
try:
    df = pd.read_csv('Data_for_UCI_named.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: File not found.")

# --- 2. PREPROCESSING ---
# Convert target 'stabf' (stable/unstable) to numbers
encoder = LabelEncoder()
df['stabf'] = encoder.fit_transform(df['stabf'])

# Select Features (Reaction Time 'tau', Power 'p', Elasticity 'g')
features = ['tau1', 'tau2', 'tau3', 'tau4', 'p1', 'p2', 'p3', 'p4', 'g1', 'g2', 'g3', 'g4']
X = df[features]
y = df['stabf']

# --- 3. SPLIT DATA ---
# 80% Training, 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. TRAIN MODEL ---
print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. EVALUATE ---
predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions) * 100

# Print Results
print(f"Model Accuracy: {acc:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, predictions))

# --- 6. VISUALIZATION ---
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Oranges')
plt.title('Grid Stability Prediction Results')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

ISTANBUL GEDIK UNIVERSITY 
MADE BY ALI IHAB ELSHATLAWY     ID 231023279
