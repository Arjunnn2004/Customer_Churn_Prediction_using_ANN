# 🔮 Customer Churn Predictor

A machine learning web application built with Streamlit that predicts customer churn using an Artificial Neural Network (ANN).

## 📋 Features

- **Interactive Web Interface**: Easy-to-use Streamlit app for making predictions
- **Real-time Predictions**: Instant churn probability calculations
- **Risk Factor Analysis**: Identifies key factors contributing to churn risk
- **Comprehensive Input Form**: All customer features can be inputted through the UI
- **Model Performance Metrics**: Displays prediction confidence and risk levels

## 🚀 Quick Start

### Option 1: Run Everything Automatically (Windows)
```bash
run_app.bat
```

### Option 2: Manual Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model**
   ```bash
   python train_model.py
   ```

3. **Run the Streamlit App**
   ```bash
   streamlit run streamlit_app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📊 Model Information

- **Algorithm**: Artificial Neural Network (ANN)
- **Architecture**: 3 layers (26 → 15 → 1 neurons)
- **Activation Functions**: ReLU (hidden), Sigmoid (output)
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-entropy
- **Data Balancing**: SMOTE (Synthetic Minority Over-sampling)
- **Feature Scaling**: MinMax Scaler

## 📁 File Structure

```
churn_predictor/
├── streamlit_app.py      # Main Streamlit application
├── train_model.py        # Model training script
├── chrun.csv            # Dataset
├── requirements.txt      # Python dependencies
├── run_app.bat          # Windows batch script
├── README.md            # This file
├── churn_model.h5       # Trained model (generated)
├── scaler.pkl           # Fitted scaler (generated)
└── feature_columns.pkl  # Feature reference (generated)
```

## 🎯 How to Use the App

1. **Customer Demographics**: Enter basic customer information
2. **Services**: Select phone and internet services
3. **Security & Support**: Choose security and support options
4. **Streaming Services**: Select streaming preferences
5. **Contract & Payment**: Enter contract and payment details
6. **Predict**: Click the prediction button to get results

## 📈 Understanding Results

- **High Risk (>70%)**: Customer likely to churn - immediate action recommended
- **Medium Risk (30-70%)**: Monitor closely and consider targeted offers
- **Low Risk (<30%)**: Customer likely to stay - continue standard service

## 🔧 Key Features Analyzed

The model considers 26 features including:
- Demographics (gender, age, dependents)
- Service subscriptions (phone, internet, streaming)
- Contract details (type, payment method)
- Financial data (monthly and total charges)
- Service add-ons (security, backup, support)

## 📊 Model Performance

The model is trained on a balanced dataset using SMOTE and achieves good performance in predicting customer churn. Training metrics are displayed during the model training process.

## 🛠️ Technical Details

### Data Preprocessing
1. Remove invalid records
2. Handle categorical variables with one-hot encoding
3. Convert binary features to 0/1
4. Scale numerical features using MinMaxScaler
5. Balance classes using SMOTE

### Model Architecture
- Input Layer: 26 features
- Hidden Layer 1: 26 neurons (ReLU)
- Hidden Layer 2: 15 neurons (ReLU)
- Output Layer: 1 neuron (Sigmoid)

## 🔄 Updating the Model

To retrain the model with new data:
1. Replace `chrun.csv` with your new dataset
2. Run `python train_model.py`
3. Restart the Streamlit app

## 🐛 Troubleshooting

### Common Issues

**"Model files not found" Error**
- Make sure you've run `train_model.py` first
- Check that `churn_model.h5` and `scaler.pkl` exist

**Import Errors**
- Install requirements: `pip install -r requirements.txt`
- Make sure you're using Python 3.8+

**CSV File Not Found**
- Ensure `chrun.csv` is in the same directory
- Check the file name spelling

## 📝 Requirements

- Python 3.8+
- TensorFlow 2.13+
- Streamlit 1.28+
- See `requirements.txt` for full dependencies

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.
