
# Disease Prediction and Symptom Severity Analysis

This project uses machine learning models to predict diseases based on symptoms, with a focus on evaluating and comparing the accuracy of different classifiers, specifically Decision Tree and Random Forest models. The dataset includes symptom severity rankings, allowing the model to make informed predictions on disease types.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Dataset Overview](#dataset-overview)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Training](#model-training)
5. [Model Testing and Evaluation](#model-testing-and-evaluation)
6. [How to Use](#how-to-use)
7. [Results and Comparison](#results-and-comparison)
8. [Requirements](#requirements)

---

### Project Structure
- `dataset.csv`: Contains raw data with diseases and corresponding symptoms.
- `Symptom-severity.csv`: Symptom severity data with weights.
- `symptom_Description.csv`: Descriptions for each disease.
- `symptom_precaution.csv`: Precautions to take for each disease.
- `random_forest.joblib`: Trained Random Forest model saved for prediction purposes.
- `README.md`: Documentation file.
- `main.py`: Main code file to run preprocessing, training, and testing models.

---

### Dataset Overview
The project uses two main datasets:
- **Symptoms Dataset**: Lists symptoms for each disease.
- **Symptom Severity Dataset**: Provides a weight ranking for each symptom to enhance prediction accuracy.

---

### Data Preprocessing
1. **Removing Hyphens**: Replaces underscores (`_`) with spaces in symptom names.
2. **Null and NaN Value Handling**: Checks for null values and fills missing values with zero.
3. **Symptom Encoding**: Encodes each symptom with a severity weight to normalize data for model training.
4. **Redundant Data Removal**: Removes columns with all zero values to avoid redundancy.

### Model Training
1. **Feature Selection**: Uses symptom columns as features and diseases as labels.
2. **Data Splitting**: Divides data into training (80%) and testing (20%) sets.
3. **Classifiers**: Trains Decision Tree and Random Forest models for comparison.
4. **Cross-Validation**: Applies k-fold cross-validation for accuracy consistency.

### Model Testing and Evaluation
Evaluates each model using:
- **Confusion Matrix**: Visualizes correct and incorrect predictions.
- **F1-Score and Accuracy**: Measures model performance.
- **Cross-Validation Results**: Provides mean accuracy and standard deviation to understand model reliability.

### Results and Comparison
A comparison plot of the algorithms, showing:
- **Training Accuracy**
- **Testing Accuracy**
- **Standard Deviation of Accuracy**

### Requirements
To set up the environment, install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key libraries used:
- `pandas`: Data manipulation
- `numpy`: Data operations
- `matplotlib`, `seaborn`: Visualization
- `scikit-learn`: Machine learning models and evaluation

### How to Use

1. **Load and Preprocess Data**:
   ```python
   df = pd.read_csv('dataset.csv')
   df1 = pd.read_csv('Symptom-severity.csv')
   ```

2. **Train Models**:
   ```python
   # Train Decision Tree
   tree = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=13)
   tree.fit(x_train, y_train)

   # Train Random Forest
   rnd_forest = RandomForestClassifier(random_state=42, max_features='sqrt', n_estimators=500, max_depth=13)
   rnd_forest.fit(x_train, y_train)
   ```

3. **Evaluate Models**:
   ```python
   # Evaluate using F1-score and accuracy
   print(f1_score(y_test, preds, average='macro'))
   print(accuracy_score(y_test, preds))
   ```

4. **Make Predictions**:
   Run the prediction function to simulate a user symptom check.
   ```python
   predd(rnd_forest, "fever", "nausea", "headache", "cough", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
   ```

5. **Saving and Loading the Model**:
   ```python
   joblib.dump(rnd_forest, "random_forest.joblib")
   loaded_rf = joblib.load("random_forest.joblib")
   ```

6. **Visualize Results**:
   Use the comparison plot to see the accuracy and reliability of each algorithm.
   ```python
   plt.show()
   ```

---

### Results
This analysis demonstrated that the **Random Forest** classifier performed better in terms of accuracy and robustness compared to **Decision Tree**. Further improvements could include tuning model parameters and testing with additional symptom datasets.

---

Feel free to contribute or suggest improvements!
