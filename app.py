"""
Amazon Sales Predictive API

Prerequisites:
If the server fails to launch, ensure you have installed the required libraries:
    pip install fastapi uvicorn

To run the server locally:
    python -m uvicorn app:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import tensorflow as tf

app = FastAPI(
    title="Amazon Sales Predictive API",
    description="A REST Endpoint for estimating E-commerce Unit Demand via XGBoost, SVC, Bidirectional LSTM, and Softmax Neural Network.",
    version="1.0"
)

# ---------------------------------------------------------
# 1. Global Objects to hold the dynamically loaded models
# ---------------------------------------------------------
preprocessor = None
lstm_scaler = None
xgb_model = None
svc_model = None
lstm_model = None
softmax_model = None
softmax_scaler = None
CATEGORY_LABELS = None

# We force the Keras loader not to error out when finding custom objects if any. 
# Though our LSTM is standard.
@app.on_event("startup")
def load_artifacts():
    global preprocessor, lstm_scaler, xgb_model, svc_model, lstm_model, softmax_model, softmax_scaler, CATEGORY_LABELS
    try:
        print("Initializing FastAPI Endpoint. Booting model instances into memory...")
        # Load the artifacts from the Jupyter execution output directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, 'models')
        
        preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
        lstm_scaler = joblib.load(os.path.join(models_dir, 'lstm_scaler.pkl'))
        
        xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_model.pkl'))
        svc_model = joblib.load(os.path.join(models_dir, 'svc_model.pkl'))
        
        lstm_model = tf.keras.models.load_model(os.path.join(models_dir, 'lstm_model.keras'))
        
        softmax_model = tf.keras.models.load_model(os.path.join(models_dir, 'softmax_model.keras'))
        softmax_scaler = joblib.load(os.path.join(models_dir, 'softmax_scaler.pkl'))
        CATEGORY_LABELS = joblib.load(os.path.join(models_dir, 'category_labels.pkl'))
        print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Could not load initial models. Please run the deployment cells in your Notebook first. Error: {e}")


# ---------------------------------------------------------
# 2. Feature column order exactly as the preprocessor was trained on
# ---------------------------------------------------------
FEATURE_COLUMNS = [
    'Product_Category', 'Price', 'Discount_Percent', 'Final_Price',
    'Ad_Spend_PPC', 'Stock_Level', 'Amazon_Buy_Box_Percentage', 'FBA_Status',
    'Day_Of_Week', 'Is_Holiday', 'Weather', 'Season'
]

# Column order for the Softmax preprocessor (all cols except Date & Product_Category)
SOFTMAX_FEATURE_COLUMNS = [
    'Price', 'Discount_Percent', 'Final_Price', 'Ad_Spend_PPC', 'Stock_Level',
    'Amazon_Buy_Box_Percentage', 'FBA_Status', 'Day_Of_Week', 'Is_Holiday',
    'Weather', 'Season', 'Units_Sold', 'High_Demand'
]

# ---------------------------------------------------------
# 3. Schema Validation Models (Expected JSON Input)
# ---------------------------------------------------------
class SingleProductTransaction(BaseModel):
    Product_Category: str
    FBA_Status: str
    Price: float
    Discount_Percent: float
    Final_Price: float
    Ad_Spend_PPC: float
    Stock_Level: int
    Amazon_Buy_Box_Percentage: float
    Weather: str
    Day_Of_Week: int
    Is_Holiday: int
    Season: int

class TimeSeriesHistory(BaseModel):
    # For LSTM we expect a flattened array of the past 7 days of raw aggregated units sold
    past_7_days_units: list[float]

class ProductCategoryInput(BaseModel):
    """
    Full feature set used by preprocessor_multi in the Extra notebook.
    All fields except Date and Product_Category from the original dataset.
    """
    Price: float
    Discount_Percent: float
    Final_Price: float
    Ad_Spend_PPC: float
    Stock_Level: int
    Amazon_Buy_Box_Percentage: float
    FBA_Status: str
    Day_Of_Week: int
    Is_Holiday: int
    Weather: str
    Season: int
    Units_Sold: int
    High_Demand: int


# ---------------------------------------------------------
# 3. Micro REST API Endpoints (Single Product Transactions)
# ---------------------------------------------------------
@app.post("/predict/units_sold")
def predict_units_sold(data: SingleProductTransaction):
    """
    Given a single JSON object containing product pricing, logistics heuristics, and temporal elements,
    utilizes the XGBoost Ensemble Regressor to spit out exactly how many units it should sell that day.
    """
    try:
        # Convert JSON into a 1-row Pandas DataFrame, reindexed to match the
        # exact column order the ColumnTransformer was trained on
        df = pd.DataFrame([data.dict()])[FEATURE_COLUMNS]
        
        # Scale/Encode data using the fitted preprocessing pipeline
        X_scaled = preprocessor.transform(df)
        
        # Predict
        prediction = xgb_model.predict(X_scaled)[0]
        
        return {
            "model_used": "XGBoost Regressor",
            "predicted_units_sold": float(round(prediction))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/high_demand")
def predict_high_demand(data: SingleProductTransaction):
    """
    Given an identical JSON input, utilizes the RBF kernel Support Vector Machine (SVC) to
    classify whether this product specific day classifies as a Logistics 'High Demand' spike.
    """
    try:
        df = pd.DataFrame([data.dict()])[FEATURE_COLUMNS]
        X_scaled = preprocessor.transform(df)
        
        prediction = svc_model.predict(X_scaled)[0]
        label = "High Demand Warning!" if prediction == 1 else "Normal Demand"
        
        return {
            "model_used": "Support Vector Classifier",
            "classification_code": int(prediction),
            "predicted_status": label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# 4. Macro REST API Endpoint (Time Series Deep Learning)
# ---------------------------------------------------------
@app.post("/predict/store_volume")
def predict_macro_volume(data: TimeSeriesHistory):
    """
    Given a JSON Array of exactly 7 days of aggregated historic volume, the Bidirectional LSTM
    Neural Network forecasts the store-wide aggregate daily volume for tomorrow.
    """
    if len(data.past_7_days_units) != 7:
        raise HTTPException(status_code=400, detail="The LSTM Architecture requires exactly 7 days of look-back history.")
    
    try:
        
        # Format the numbers for the neural network scaling boundaries (0,1)
        raw_seq = np.array(data.past_7_days_units).reshape(-1, 1)
        scaled_seq = lstm_scaler.transform(raw_seq)
        
        # LSTMs require 3D arrays: (Batch_Size=1, Timesteps=7, Features=1)
        lstm_input = scaled_seq.reshape((1, 7, 1))
        
        prediction = lstm_model.predict(lstm_input)
        
        # Un-scale to get standard volume
        absolute_forecast = lstm_scaler.inverse_transform(prediction)[0][0]
        
        return {
            "model_used": "Bidirectional LSTM",
            "predicted_store_aggregate_volume_tomorrow": float(round(absolute_forecast))
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------
# 5. Multi-Class REST API Endpoint (Softmax Category Classifier)
# ---------------------------------------------------------
@app.post("/predict/product_category")
def predict_product_category(data: ProductCategoryInput):
    """
    Given the full product transaction features (all columns except Date and
    Product_Category), the Softmax Neural Network classifies the macro
    Product_Category of the item. Returns the predicted category, confidence
    score, and probability distribution across all classes.
    """
    try:
        df = pd.DataFrame([data.dict()])[SOFTMAX_FEATURE_COLUMNS]
        scaled_features = softmax_scaler.transform(df)
        
        probabilities = softmax_model.predict(scaled_features)[0]
        predicted_index = int(np.argmax(probabilities))
        predicted_label = CATEGORY_LABELS[predicted_index]
        confidence = float(round(probabilities[predicted_index] * 100, 2))
        
        return {
            "model_used": "Softmax Neural Network",
            "predicted_category": predicted_label,
            "confidence_percent": confidence,
            "all_class_probabilities": {
                CATEGORY_LABELS[i]: float(round(p * 100, 2))
                for i, p in enumerate(probabilities)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"message": "Amazon Forecast API is active. Go to /docs to test the interactive REST endpoints!"}
