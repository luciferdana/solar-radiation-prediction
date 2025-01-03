import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Load data
df = pd.read_csv('SolarPrediction.csv')

# Prepare features and target
X = df[['Temperature', 'Pressure', 'Humidity', 'WindDirection(Degrees)', 'Speed']]
y = df['Radiation']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
cv_scores = cross_val_score(model, X, y, cv=5)
cv_mean = cv_scores.mean()

# Save metrics
model_info = {
    'r2_score': r2,
    'mae': mae,
    'rmse': rmse,
    'cv_score': cv_mean,
    'cv_scores': cv_scores.tolist(),
    'feature_importance': dict(zip(X.columns, model.feature_importances_))
}

print(f"Model Performance Metrics:")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"Cross-validation Score: {cv_mean:.4f}")
print("\nFeature Importance:")
for feature, importance in model_info['feature_importance'].items():
    print(f"{feature}: {importance:.4f}")

# Save model and metrics
joblib.dump(model_info, 'model_info.joblib')