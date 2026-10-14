import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

# 1. Generate or Load Traffic Time-Series Data
def generate_sample_data(samples=1000):
    time_steps = np.linspace(0, 100, samples)
    base_traffic = 15 + 10 * np.sin(time_steps / 5) + np.random.normal(0, 2, samples)
    vehicle_counts = np.clip(base_traffic, 0, 35).astype(int)
    return pd.DataFrame({'vehicle_count': vehicle_counts})

# 2. Feature Engineering: Create Lag Features
def create_features(df, lag_steps=5, forecast_horizon=1):
    data = df.copy()
    for i in range(1, lag_steps + 1):
        data[f'lag_{i}'] = data['vehicle_count'].shift(i)
    
    data['target'] = data['vehicle_count'].shift(-forecast_horizon)
    data = data.dropna()
    
    feature_cols = [f'lag_{i}' for i in range(1, lag_steps + 1)]
    X = data[feature_cols]
    y = data['target']
    
    return X, y

if __name__ == "__main__":
    df = generate_sample_data(1000)
    
    # Generate lag features (5 past frames -> predict 1 frame ahead)
    X, y = create_features(df, lag_steps=5, forecast_horizon=1)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 3. Train XGBoost Regressor
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    print("--- XGBoost Training Report ---")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, preds):.2f} vehicles")
    print(f"R2 Score: {r2_score(y_test, preds):.2f}")
    
    # Save trained model to disk
    joblib.dump(model, "traffic_xgb_model.pkl")
    print("[INFO] Model saved as traffic_xgb_model.pkl")
