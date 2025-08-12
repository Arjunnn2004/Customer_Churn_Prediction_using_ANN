"""
Train and save the churn prediction model for Streamlit app
Run this script first before using the Streamlit app
"""

import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(df):
    """Preprocess the data following the same steps as in the notebook"""
    # Drop customerID
    df = df.drop('customerID', axis='columns')
    
    # Remove rows with empty TotalCharges
    df = df[df['TotalCharges'] != ' ']
    
    # Replace service values
    df = df.replace('No internet service', 'No')
    df = df.replace('No phone service', 'No')
    
    # Convert Yes/No columns to 1/0
    yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies', 
                      'PaperlessBilling', 'Churn']
    for col in yes_no_columns:
        df[col] = df[col].replace({'Yes': 1, 'No': 0})
    
    # Convert gender
    df['gender'] = df['gender'].replace({'Female': 1, 'Male': 0})
    
    # Create dummy variables
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])
    
    # Convert boolean columns to int
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    
    # Ensure all numeric columns are float type
    numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert entire dataframe to float32 for TensorFlow compatibility
    df = df.astype(np.float32)
    
    return df

def create_ann_model(input_dim):
    """Create the ANN model with the same architecture as in the notebook"""
    model = keras.Sequential([
        keras.layers.Dense(26, input_dim=input_dim, activation='relu'),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("🚀 Starting model training...")
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    df = pd.read_csv('chrun.csv')
    df_processed = preprocess_data(df)
    
    print(f"✅ Data shape after preprocessing: {df_processed.shape}")
    
    # Scale numerical features
    print("🔧 Scaling numerical features...")
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = MinMaxScaler()
    df_processed[cols_to_scale] = scaler.fit_transform(df_processed[cols_to_scale])
    
    # Prepare features and target
    X = df_processed.drop('Churn', axis='columns')
    y = df_processed['Churn']
    
    print(f"📈 Feature columns: {X.columns.tolist()}")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    
    # Apply SMOTE for balancing
    print("⚖️ Applying SMOTE for class balancing...")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_sm, y_sm = smote.fit_resample(X, y)
    
    print(f"📊 Balanced target distribution: {pd.Series(y_sm).value_counts().to_dict()}")
    
    # Split the data
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_sm, y_sm, test_size=0.2, random_state=5, stratify=y_sm
    )
    
    # Create and train model
    print("🤖 Creating and training ANN model...")
    model = create_ann_model(X_train.shape[1])
    
    print("🏋️ Training model (this may take a few minutes)...")
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Evaluate model
    print("📊 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")
    print(f"✅ Test Loss: {test_loss:.4f}")
    
    # Make predictions for classification report
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.round(y_pred_prob)
    
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model and scaler
    print("💾 Saving model and scaler...")
    model.save('churn_model.h5')
    
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature columns for reference
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    print("✅ Model and scaler saved successfully!")
    print("\n🎉 Training complete! You can now run the Streamlit app.")
    print("💡 Run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()
