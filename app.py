from flask import Flask, request, jsonify
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def read_csv_file(file_path):
    """Attempts to read a CSV file using UTF-8 first, then falls back to latin1."""
    try:
        return pd.read_csv(file_path)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(file_path, encoding="latin1")
        except Exception as e:
            raise Exception("Failed to decode CSV file: " + str(e))

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles CSV file upload and returns column names."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    try:
        df = read_csv_file(file_path)
        df = df.dropna(how='all')  # Drop completely empty rows
        return jsonify({"columns": df.columns.tolist(), "message": "File uploaded successfully", "filename": file.filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train_model():
    """Trains multiple regression models and returns accuracy."""
    data = request.json
    file_path = os.path.join(UPLOAD_FOLDER, data["filename"])
    
    try:
        df = read_csv_file(file_path)
        input_cols = data.get("input_cols", [])
        output_col = data.get("output_col", "")

        if not input_cols or not output_col:
            return jsonify({"error": "Please provide valid input and output columns."}), 400
        
        if output_col not in df.columns or any(col not in df.columns for col in input_cols):
            return jsonify({"error": "Selected columns do not exist in the dataset."}), 400

        X = df[input_cols]
        y = df[output_col]

        if y.dtype == "object":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)

        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in numerical_features:
            X[col] = X[col].fillna(X[col].median())
        for col in categorical_features:
            X[col] = X[col].fillna(X[col].mode()[0])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        test_size = float(data.get("test_size", 0.2))
        min_test_samples = 2
        if len(y) * test_size < min_test_samples:
            test_size = min_test_samples / len(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if len(y_test) < 2:
            return jsonify({"error": "Not enough test samples. Try increasing the dataset size or adjusting test_size."}), 400

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "SVR": SVR(),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
        }

        results = {}
        for name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, np.round(predictions)) if len(y_test) > 1 else "N/A"

            results[name] = {"Accuracy": round(accuracy, 4) if isinstance(accuracy, float) else accuracy}

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=10000)