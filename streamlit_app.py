import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Load the trained model and scaler
@st.cache_resource
def load_model_and_scaler():
    try:
        model = keras.models.load_model('churn_model.h5')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

def preprocess_input(data):
    """Preprocess user input to match training data format"""
    df = pd.DataFrame([data])
    
    # Replace service values
    df = df.replace('No internet service', 'No')
    df = df.replace('No phone service', 'No')
    
    # Convert Yes/No columns to 1/0
    yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                      'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                      'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    for col in yes_no_columns:
        df[col] = df[col].replace({'Yes': 1, 'No': 0})
    
    # Convert gender
    df['gender'] = df['gender'].replace({'Female': 1, 'Male': 0})
    
    # Ensure numerical columns are properly typed
    df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce')
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Create dummy variables for categorical columns
    df = pd.get_dummies(df, columns=['InternetService', 'Contract', 'PaymentMethod'])
    
    # Convert all boolean columns to integers
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    
    # Ensure all expected columns are present
    expected_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
        'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    # Add missing columns with default value 0
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[expected_columns]
    
    # Convert all columns to float32 for consistency
    df = df.astype(np.float32)
    
    return df

def main():
    st.set_page_config(
        page_title="Customer Churn Predictor",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔮 Customer Churn Prediction App")
    st.markdown("---")
    
    # Load model and scaler
    model, scaler = load_model_and_scaler()
    
    if model is None:
        st.error("⚠️ Model files not found! Please train the model first by running the notebook.")
        st.info("💡 Run the Jupyter notebook to train and save the model, then refresh this page.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👤 Customer Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1], 
                                    help="0 = No, 1 = Yes")
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        
        st.subheader("📞 Services")
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", 
                                    ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", 
                                      ["DSL", "Fiber optic", "No"])
        
    with col2:
        st.subheader("🔒 Security & Support")
        online_security = st.selectbox("Online Security", 
                                     ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", 
                                   ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", 
                                       ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", 
                                  ["No", "Yes", "No internet service"])
        
        st.subheader("📺 Streaming Services")
        streaming_tv = st.selectbox("Streaming TV", 
                                  ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", 
                                      ["No", "Yes", "No internet service"])
        
    # Contract and Payment Information
    st.subheader("💳 Contract & Payment")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        contract = st.selectbox("Contract Type", 
                              ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    
    with col4:
        payment_method = st.selectbox("Payment Method", 
                                    ["Electronic check", "Mailed check", 
                                     "Bank transfer (automatic)", 
                                     "Credit card (automatic)"])
    
    with col5:
        monthly_charges = st.number_input("Monthly Charges ($)", 
                                        min_value=0.0, max_value=200.0, 
                                        value=50.0, step=0.01)
        total_charges = st.number_input("Total Charges ($)", 
                                      min_value=0.0, max_value=10000.0, 
                                      value=500.0, step=0.01)
    
    # Prediction button
    if st.button("🔮 Predict Churn", type="primary"):
        # Prepare input data
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        try:
            # Preprocess the input
            processed_data = preprocess_input(input_data)
            
            # Debug information
            st.write("Debug Info:")
            st.write(f"Processed data shape: {processed_data.shape}")
            st.write(f"Data types: {processed_data.dtypes.to_dict()}")
            
            # Scale the numerical features
            cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
            processed_data[cols_to_scale] = scaler.transform(processed_data[cols_to_scale])
            
            # Ensure all data is float32 for TensorFlow compatibility
            processed_data = processed_data.astype(np.float32)
            
            # Make prediction
            prediction_array = processed_data.values.reshape(1, -1)
            st.write(f"Input array shape: {prediction_array.shape}")
            st.write(f"Input array dtype: {prediction_array.dtype}")
            
            prediction_prob = model.predict(prediction_array, verbose=0)[0][0]
            prediction = 1 if prediction_prob > 0.5 else 0
            
            # Display results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if prediction == 1:
                    st.error("🚨 **HIGH RISK**: Customer likely to churn")
                    st.markdown(f"**Churn Probability:** {prediction_prob:.2%}")
                else:
                    st.success("✅ **LOW RISK**: Customer likely to stay")
                    st.markdown(f"**Retention Probability:** {(1-prediction_prob):.2%}")
            
            with col_res2:
                # Create a gauge-like visualization
                st.metric(
                    label="Churn Risk Score", 
                    value=f"{prediction_prob:.2%}",
                    delta=f"{'High' if prediction_prob > 0.7 else 'Medium' if prediction_prob > 0.3 else 'Low'} Risk"
                )
            
            # Risk factors analysis
            st.subheader("📊 Risk Factor Analysis")
            risk_factors = []
            
            if contract == "Month-to-month":
                risk_factors.append("Month-to-month contract (higher churn risk)")
            if payment_method == "Electronic check":
                risk_factors.append("Electronic check payment (higher churn risk)")
            if tenure < 12:
                risk_factors.append("Low tenure (new customers have higher churn risk)")
            if monthly_charges > 70:
                risk_factors.append("High monthly charges")
            if internet_service == "Fiber optic" and online_security == "No":
                risk_factors.append("Fiber optic without security services")
            
            if risk_factors:
                st.warning("⚠️ **Key Risk Factors:**")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.info("ℹ️ No major risk factors identified")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check your input values and try again.")
    
    # Add information about the model
    st.markdown("---")
    with st.expander("ℹ️ About this Model"):
        st.markdown("""
        **Model Information:**
        - **Algorithm:** Artificial Neural Network (ANN)
        - **Architecture:** 3 layers (26 → 15 → 1 neurons)
        - **Activation:** ReLU (hidden layers), Sigmoid (output)
        - **Training:** Binary cross-entropy loss, Adam optimizer
        - **Data Processing:** SMOTE for class balancing, MinMax scaling
        
        **How to interpret results:**
        - **High Risk (>70%):** Immediate retention actions recommended
        - **Medium Risk (30-70%):** Monitor closely, consider targeted offers
        - **Low Risk (<30%):** Continue standard customer service
        """)

if __name__ == "__main__":
    main()
