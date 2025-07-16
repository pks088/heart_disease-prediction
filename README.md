🫀 Heart Disease Prediction using Machine Learning

This project aims to predict the presence of heart disease in a patient using various supervised machine learning algorithms. The prediction is based on multiple clinical parameters such as age, chest pain type, blood pressure, cholesterol level, and more.

📁 Project Structure
`Heart_disease.ipynb`: Jupyter Notebook containing all code, data preprocessing, model training, evaluation, and comparison.

🧠 Algorithms Used

- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest

📊 Dataset

The dataset used is from the **UCI Heart Disease Repository**. It contains features like:
- Age
- Sex
- Chest pain type (cp)
- Resting blood pressure (trestbps)
- Serum cholesterol (chol)
- Fasting blood sugar (fbs)
- Resting electrocardiographic results (restecg)
- Maximum heart rate achieved (thalach)
- Exercise induced angina (exang)
- ST depression (oldpeak)
- Number of major vessels (ca)
- Thalassemia (thal)
- Target (1 = heart disease, 0 = no heart disease)

⚙️ Workflow

1. **Data Preprocessing**: Handled missing values and categorical variables.
2. **Feature Selection**: Used correlation heatmaps and domain knowledge.
3. **Model Training**: Trained four different ML models with hyperparameter tuning.
4. **Evaluation**: Compared models based on accuracy, confusion matrix, and classification report.

🏆 Results

| Model              | Accuracy |
|-------------------|----------|
| Logistic Regression | 85.78%   |
| SVM                 | 83.82%    |
| KNN                 | 85.29%    |
| Random Forest       | 89.7%    |


🛠️ Libraries Used

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
