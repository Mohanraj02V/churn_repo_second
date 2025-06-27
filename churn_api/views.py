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

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Fixed path to go up one level

# Load model artifacts
model = joblib.load(os.path.join(BASE_DIR, 'churn_api/model/xgb_churn_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'churn_api/model/scaler.pkl'))
encoder = joblib.load(os.path.join(BASE_DIR, 'churn_api/model/encoder.pkl'))
imputer = joblib.load(os.path.join(BASE_DIR, 'churn_api/model/imputer.pkl'))
numeric_features = joblib.load(os.path.join(BASE_DIR, 'churn_api/model/numeric_features.pkl'))
categorical_features = joblib.load(os.path.join(BASE_DIR, 'churn_api/model/categorical_features.pkl'))
encoded_cols = joblib.load(os.path.join(BASE_DIR, 'churn_api/model/encoded_columns.pkl'))

class PredictChurnAPIView(APIView):
    def post(self, request):
        try:
            input_data = request.data
            
            # Validate required fields exist
            required_fields = ['TotalCharges', 'tenure', 'MonthlyCharges'] + categorical_features.tolist()
            missing_fields = [field for field in required_fields if field not in input_data]
            if missing_fields:
                return Response(
                    {"error": f"Missing required fields: {missing_fields}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Create DataFrame with explicit numeric conversion
            input_df = pd.DataFrame([input_data])
            
            # Convert numeric fields to float
            for field in ['TotalCharges', 'tenure', 'MonthlyCharges']:
                input_df[field] = pd.to_numeric(input_df[field], errors='coerce')
                if pd.isna(input_df[field]).any():
                    return Response(
                        {"error": f"Invalid numeric value for {field}"},
                        status=status.HTTP_400_BAD_REQUEST
                    )

            # Feature Engineering with division protection
            input_df['Avg_Monthly_Charge'] = input_df['TotalCharges'] / (input_df['tenure'].replace(0, 1))  # Avoid division by zero

            # Filter categorical features
            filtered_categorical_features = [col for col in categorical_features if col not in ['customerID', 'Churn']]

            # Impute, Scale, Encode
            input_df[numeric_features] = imputer.transform(input_df[numeric_features])
            input_df[numeric_features] = scaler.transform(input_df[numeric_features])
            input_df[encoded_cols] = encoder.transform(input_df[filtered_categorical_features])
            
            # Make prediction
            x_input = input_df[numeric_features + encoded_cols]
            pred = model.predict(x_input)[0]
            proba = model.predict_proba(x_input)[0][1]
            
            result = {
                "prediction": "Yes" if pred == 1 else "No",
                "churn_probability": round(float(proba) * 100, 2)  # Explicit float conversion
            }
            return Response(result, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
            return Response({"error": f"Invalid data format: {str(e)}"}, 
                          status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": f"Prediction failed: {str(e)}"}, 
                          status=status.HTTP_500_INTERNAL_SERVER_ERROR)
