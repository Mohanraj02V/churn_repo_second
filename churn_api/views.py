'''
import joblib
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
'''
'''
import os
import joblib
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model artifacts
model = joblib.load(os.path.join(BASE_DIR, 'model/xgb_churn_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'model/scaler.pkl'))
encoder = joblib.load(os.path.join(BASE_DIR, 'model/encoder.pkl'))
imputer = joblib.load(os.path.join(BASE_DIR, 'model/imputer.pkl'))
numeric_features = joblib.load(os.path.join(BASE_DIR, 'model/numeric_features.pkl'))
categorical_features = joblib.load(os.path.join(BASE_DIR, 'model/categorical_features.pkl'))
encoded_cols = joblib.load(os.path.join(BASE_DIR, 'model/encoded_columns.pkl'))

filtered_categorical_features = [col for col in categorical_features if col not in ['customerID', 'Churn']]

class PredictChurnAPIView(APIView):
    def post(self, request):
        try:
            input_data = request.data
            input_df = pd.DataFrame([input_data])

            # Ensure required columns are present
            for col in ['TotalCharges', 'tenure']:
                if col not in input_df.columns:
                    return Response({"error": f"Missing required field: '{col}'"}, status=status.HTTP_400_BAD_REQUEST)

            # Convert TotalCharges to numeric safely
            input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce')
            input_df['tenure'] = pd.to_numeric(input_df['tenure'], errors='coerce')

            # Check for NaNs after conversion
            if input_df['TotalCharges'].isnull().any() or input_df['tenure'].isnull().any():
                return Response({"error": "TotalCharges and tenure must be valid numbers"}, status=status.HTTP_400_BAD_REQUEST)

            # Feature Engineering
            input_df['Avg_Monthly_Charge'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)

            # Impute, Scale, Encode
            input_df[numeric_features] = imputer.transform(input_df[numeric_features])
            input_df[numeric_features] = scaler.transform(input_df[numeric_features])
            encoded_array = encoder.transform(input_df[filtered_categorical_features])
            encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

            input_df = pd.concat([input_df[numeric_features].reset_index(drop=True),
                                  encoded_df.reset_index(drop=True)], axis=1)

            x_input = input_df[numeric_features + encoded_cols]

            pred = model.predict(x_input)[0]
            proba = model.predict_proba(x_input)[0][1]

            return Response({
                "prediction": "Yes" if pred == 1 else "No",
                "churn_probability": round(proba * 100, 2)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
'''
class PredictChurnAPIView(APIView):
    def post(self, request):
        try:
            input_data = request.data

            # Validate required fields
            required_fields = ['tenure', 'MonthlyCharges', 'TotalCharges']
            for field in required_fields:
                if field not in input_data:
                    return Response({"error": f"Missing required field: '{field}'"}, status=status.HTTP_400_BAD_REQUEST)

            input_df = pd.DataFrame([input_data])

            # Handle any missing or null TotalCharges
            if pd.isnull(input_df['TotalCharges'].iloc[0]) or input_df['TotalCharges'].iloc[0] == "":
                input_df['TotalCharges'] = 0.0
            else:
                input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0.0)

            # Avg_Monthly_Charge feature
            input_df['Avg_Monthly_Charge'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)

            # Filter categorical features
            filtered_categorical_features = [col for col in categorical_features if col not in ['customerID', 'Churn']]

            # Impute → Scale → Encode
            input_df[numeric_features] = imputer.transform(input_df[numeric_features])
            input_df[numeric_features] = scaler.transform(input_df[numeric_features])
            encoded = encoder.transform(input_df[filtered_categorical_features])
            encoded_df = pd.DataFrame(encoded, columns=encoded_cols)
            input_df.reset_index(drop=True, inplace=True)
            final_df = pd.concat([input_df[numeric_features], encoded_df], axis=1)

            x_input = final_df
            pred = model.predict(x_input)[0]
            proba = model.predict_proba(x_input)[0][1]

            return Response({
                "prediction": "Yes" if pred == 1 else "No",
                "churn_probability": round(proba * 100, 2)
            }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
