# Customer Churn Prediction using Artificial Neural Networks (ANN)

A machine learning project that predicts customer churn using an Artificial Neural Network built with TensorFlow/Keras. This project analyzes customer behavior patterns to identify customers likely to discontinue their service.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Results](#results)
- [File Structure](#file-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)

## 🔍 Overview

Customer churn prediction is a critical business problem where companies aim to identify customers who are likely to stop using their services. This project uses deep learning techniques to build a predictive model that can help businesses:

- Identify at-risk customers
- Implement retention strategies
- Reduce customer acquisition costs
- Improve customer lifetime value

## ✨ Features

- **Deep Learning Model**: 3-layer neural network with ReLU and sigmoid activations
- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Class Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique) implementation
- **Feature Scaling**: MinMax scaling for numerical features
- **One-Hot Encoding**: Categorical variable transformation
- **Model Evaluation**: Accuracy metrics and classification reports
- **Flexible Architecture**: Configurable loss functions and class weights

## 📊 Dataset

The project uses a telecommunications customer churn dataset (`chrun.csv`) with the following characteristics:

### Original Features:
- **Customer Demographics**: CustomerID, Gender, Senior Citizen status
- **Account Information**: Tenure, Contract type, Payment method
- **Service Details**: Phone service, Internet service, Multiple lines
- **Add-on Services**: Online security, Backup, Device protection, Tech support
- **Streaming Services**: Streaming TV, Streaming movies
- **Billing Information**: Monthly charges, Total charges, Paperless billing
- **Target Variable**: Churn (Yes/No)

### Data Preprocessing Steps:
1. **Data Cleaning**: Removal of customerID and handling of missing values
2. **Data Type Conversion**: Converting TotalCharges to numeric format
3. **Missing Value Treatment**: Filtering out incomplete records
4. **Categorical Encoding**: Binary encoding for Yes/No variables
5. **One-Hot Encoding**: For multi-category variables (InternetService, Contract, PaymentMethod)
6. **Feature Scaling**: MinMax scaling for numerical features (tenure, MonthlyCharges, TotalCharges)

## 🏗️ Model Architecture

### Neural Network Structure:
```
Input Layer: 26 features
├── Hidden Layer 1: 26 neurons (ReLU activation)
├── Hidden Layer 2: 15 neurons (ReLU activation)
└── Output Layer: 1 neuron (Sigmoid activation)
```

### Model Configuration:
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Training Epochs**: 100
- **Class Imbalance**: Handled using SMOTE oversampling

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab

### Required Packages
```bash
pip install tensorflow pandas numpy scikit-learn imbalanced-learn
```

### Clone and Setup
```bash
git clone <repository-url>
cd churn_predictor
```

## 💻 Usage

1. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook churn_using_ANN.ipynb
   ```

2. **Ensure the dataset is available:**
   - Place `chrun.csv` in the same directory as the notebook

3. **Run the cells sequentially:**
   - Data loading and exploration
   - Data preprocessing and cleaning
   - Feature engineering
   - Model training and evaluation

4. **Example Usage:**
   ```python
   # Train the model
   y_preds = ANN(X_train, y_train, X_test, y_test, 'binary_crossentropy', -1)
   
   # View predictions
   print("Predictions:", y_preds[:10])
   print("Actual values:", y_test[:10])
   ```

## 🔧 Data Preprocessing

### 1. Data Cleaning
```python
# Remove customer ID
df.drop('customerID', axis='columns', inplace=True)

# Handle missing values in TotalCharges
df1 = df[df['TotalCharges'] != ' ']
```

### 2. Feature Engineering
```python
# Replace inconsistent values
df1 = df1.replace('No internet service', 'No')
df1 = df1.replace('No phone service', 'No')

# Binary encoding for Yes/No columns
yes_no_columns = ['Partner', 'Dependents', 'PhoneService', ...]
for col in yes_no_columns:
    df1[col] = df1[col].replace({'Yes': 1, 'No': 0})
```

### 3. Categorical Encoding
```python
# One-hot encoding for categorical variables
df2 = pd.get_dummies(df1, columns=['InternetService', 'Contract', 'PaymentMethod'])
```

### 4. Feature Scaling
```python
# Scale numerical features
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])
```

### 5. Class Imbalance Handling
```python
# Apply SMOTE for oversampling minority class
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X, y)
```

## 🤖 Model Training

### ANN Function
The custom `ANN` function provides a flexible interface for model training:

```python
def ANN(X_train, y_train, X_test, y_test, loss, weights):
    model = keras.Sequential([
        keras.layers.Dense(26, input_dim=26, activation='relu'),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    
    if weights == -1:
        model.fit(X_train, y_train, epochs=100)
    else:
        model.fit(X_train, y_train, epochs=100, class_weight=weights)
    
    # Evaluation and predictions
    print(model.evaluate(X_test, y_test))
    y_preds = model.predict(X_test)
    y_preds = np.round(y_preds)
    
    print("Classification Report: \n", classification_report(y_test, y_preds))
    return y_preds
```

## 📈 Results

The model provides:
- **Accuracy Metrics**: Training and testing accuracy
- **Classification Report**: Precision, Recall, F1-score for both classes
- **Confusion Matrix**: Detailed performance breakdown
- **Binary Predictions**: Rounded probability outputs (0 or 1)

### Model Performance Features:
- Handles class imbalance through SMOTE
- Provides detailed evaluation metrics
- Returns binary predictions for business decision-making

## 📁 File Structure

```
churn_predictor/
├── churn_using_ANN.ipynb    # Main Jupyter notebook
├── chrun.csv               # Customer churn dataset
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies (if created)
```

## 📦 Dependencies

- **TensorFlow/Keras**: Deep learning framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **Imbalanced-learn**: SMOTE implementation for class imbalance
- **Jupyter**: Interactive development environment

## 🔍 Key Features

### Data Processing Pipeline:
1. **Data Loading**: CSV file reading with pandas
2. **Data Cleaning**: Missing value handling and data type conversion
3. **Feature Engineering**: Categorical encoding and feature scaling
4. **Class Balancing**: SMOTE oversampling technique
5. **Train-Test Split**: 80-20 split for model validation

### Model Features:
- **Multi-layer Architecture**: Deep neural network with hidden layers
- **Activation Functions**: ReLU for hidden layers, Sigmoid for output
- **Optimization**: Adam optimizer for efficient training
- **Evaluation**: Comprehensive performance metrics

## 🎯 Business Impact

This churn prediction model can help businesses:
- **Reduce Customer Churn**: Identify at-risk customers early
- **Improve Retention**: Target specific customers with retention campaigns
- **Optimize Marketing**: Focus resources on high-value customers
- **Increase Revenue**: Reduce customer acquisition costs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Notes

- The model uses SMOTE to handle class imbalance in the dataset
- Feature scaling is applied to numerical variables for better convergence
- The model architecture can be modified by adjusting the ANN function parameters
- Binary crossentropy loss is used for binary classification

## 🏷️ Tags

`machine-learning` `deep-learning` `neural-networks` `customer-churn` `tensorflow` `keras` `classification` `smote` `data-preprocessing` `business-analytics`

---

**Author**: [Your Name]  
**Date**: August 2025  
**Version**: 1.0
