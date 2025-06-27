"""import joblib
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load model artifacts correctly from model/ folder
model = joblib.load(os.path.join(BASE_DIR, 'model/xgb_churn_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'model/scaler.pkl'))
encoder = joblib.load(os.path.join(BASE_DIR, 'model/encoder.pkl'))
imputer = joblib.load(os.path.join(BASE_DIR, 'model/imputer.pkl'))
numeric_features = joblib.load(os.path.join(BASE_DIR, 'model/numeric_features.pkl'))
categorical_features = joblib.load(os.path.join(BASE_DIR, 'model/categorical_features.pkl'))
encoded_cols = joblib.load(os.path.join(BASE_DIR, 'model/encoded_columns.pkl'))

class PredictChurnAPIView(APIView):
    def post(self, request):
        try:
            input_data = request.data
            input_df = pd.DataFrame([input_data])
            
            # Filter categorical features (use the loaded variable directly)
            filtered_categorical_features = [col for col in categorical_features if col not in ['customerID', 'Churn']]

            # Feature Engineering
            input_df['Avg_Monthly_Charge'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)

            # Impute, Scale, Encode
            input_df[numeric_features] = imputer.transform(input_df[numeric_features])
            input_df[numeric_features] = scaler.transform(input_df[numeric_features])
            input_df[encoded_cols] = encoder.transform(input_df[filtered_categorical_features])
            
            x_input = input_df[numeric_features + encoded_cols]
            pred = model.predict(x_input)[0]
            proba = model.predict_proba(x_input)[0][1]
            result = {
                "prediction": "Yes" if pred == 1 else "No",
                "churn_probability": round(proba * 100, 2)
            }
            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)




# ---- Usage ----
# Run server:
# python manage.py runserver
# Then POST to http://127.0.0.1:8000/api/predict/ with a JSON body like:
# {
#   "gender": "Female",
#   "SeniorCitizen": 0,
#   "Partner": "Yes",
#   "Dependents": "No",
#   "tenure": 5,
#   "PhoneService": "Yes",
#   "MultipleLines": "No",
#   "InternetService": "Fiber optic",
#   "OnlineSecurity": "No",
#   "OnlineBackup": "Yes",
#   "DeviceProtection": "Yes",
#   "TechSupport": "No",
#   "StreamingTV": "No",
#   "StreamingMovies": "Yes",
#   "Contract": "Month-to-month",
#   "PaperlessBilling": "Yes",
#   "PaymentMethod": "Electronic check",
#   "MonthlyCharges": 80.35,
#   "TotalCharges": 401.75
# }
"""
import joblib
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Fixed path to go up one level

class PredictChurnAPIView(APIView):
    '''
    # Load model artifacts once when the class is loaded
    MODEL_PATH = os.path.join(BASE_DIR, 'model/xgb_churn_model.pkl')
    SCALER_PATH = os.path.join(BASE_DIR, 'model/scaler.pkl')
    ENCODER_PATH = os.path.join(BASE_DIR, 'model/encoder.pkl')
    IMPUTER_PATH = os.path.join(BASE_DIR, 'model/imputer.pkl')
    NUMERIC_FEATURES_PATH = os.path.join(BASE_DIR, 'model/numeric_features.pkl')
    CATEGORICAL_FEATURES_PATH = os.path.join(BASE_DIR, 'model/categorical_features.pkl')
    ENCODED_COLS_PATH = os.path.join(BASE_DIR, 'model/encoded_columns.pkl')
    '''

    try:
        model = joblib.load(os.path.join(BASE_DIR, 'model/xgb_churn_model.pkl'))
        scaler = joblib.load(os.path.join(BASE_DIR, 'model/scaler.pkl'))
        encoder = joblib.load(os.path.join(BASE_DIR, 'model/encoder.pkl'))
        imputer = joblib.load(os.path.join(BASE_DIR, 'model/imputer.pkl'))
        numeric_features = joblib.load(os.path.join(BASE_DIR, 'model/numeric_features.pkl'))
        categorical_features = joblib.load(os.path.join(BASE_DIR, 'model/categorical_features.pkl'))
        encoded_cols = joblib.load(os.path.join(BASE_DIR, 'model/encoded_columns.pkl'))
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts: {str(e)}")

    REQUIRED_FIELDS = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges'
    ]

    def post(self, request):
        try:
            # 1. Validate input data
            input_data = request.data
            missing_fields = [field for field in self.REQUIRED_FIELDS if field not in input_data]
            if missing_fields:
                return Response(
                    {"error": f"Missing required fields: {missing_fields}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # 2. Create DataFrame with proper type conversion
            input_df = pd.DataFrame([input_data])
            
            # Convert numeric fields explicitly
            numeric_fields = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            for field in numeric_fields:
                input_df[field] = pd.to_numeric(input_df[field], errors='coerce')

            # 3. Feature Engineering with safety checks
            input_df['Avg_Monthly_Charge'] = np.where(
                input_df['tenure'] > 0,
                input_df['TotalCharges'] / input_df['tenure'],
                input_df['MonthlyCharges']  # Fallback for zero tenure
            )

            # 4. Handle categorical features
            filtered_categorical_features = [col for col in self.categorical_features 
                                          if col not in ['customerID', 'Churn']]

            # 5. Preprocessing pipeline
            # Impute missing values
            input_df[self.numeric_features] = self.imputer.transform(input_df[self.numeric_features])
            
            # Scale numeric features
            input_df[self.numeric_features] = self.scaler.transform(input_df[self.numeric_features])
            
            # Encode categorical features
            encoded_features = self.encoder.transform(input_df[filtered_categorical_features])
            input_df[self.encoded_cols] = encoded_features

            # 6. Make prediction
            x_input = input_df[self.numeric_features + self.encoded_cols]
            pred = self.model.predict(x_input)[0]
            proba = self.model.predict_proba(x_input)[0][1]
            
            result = {
                "prediction": "Yes" if pred == 1 else "No",
                "churn_probability": round(float(proba) * 100, 2),  # Explicit float conversion
                "status": "success"
            }
            return Response(result, status=status.HTTP_200_OK)

        except KeyError as e:
            return Response({"error": f"Missing field in input data: {str(e)}"}, 
                          status=status.HTTP_400_BAD_REQUEST)
        except ValueError as e:
            return Response({"error": f"Invalid data format: {str(e)}"}, 
                          status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": f"Prediction failed: {str(e)}"}, 
                          status=status.HTTP_500_INTERNAL_SERVER_ERROR)
