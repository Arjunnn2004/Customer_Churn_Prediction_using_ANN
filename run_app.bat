@echo off
echo 🚀 Customer Churn Predictor Setup
echo ================================
echo.

echo Installing required packages...
pip install -r requirements.txt

echo.
echo Training the model...
python train_model.py

echo.
echo Starting Streamlit app...
echo Open your browser to http://localhost:8501
streamlit run streamlit_app.py
