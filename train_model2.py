import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# Load data
print("Loading data...")
data = pd.read_csv('SolarPrediction.csv')

def preprocess_data(df):
    df = df.copy()
    
    # Drop unnecessary columns
    if 'Data' in df.columns:
        df.drop('Data', axis=1, inplace=True)
    
    # Convert time columns
    time_columns = ['Time', 'TimeSunRise', 'TimeSunSet']
    for col in time_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            
    # Extract time features
    if 'Time' in df.columns:
        df['Time_Hour'] = df['Time'].dt.hour
        df['Time_Minute'] = df['Time'].dt.minute
        df['Time_Second'] = df['Time'].dt.second
        
    # Format column names
    df.rename(columns={
        'WindDirection(Degrees)': 'WindDirection',
        'UNIXTime': 'unix_time'
    }, inplace=True)
    
    # Select features for model
    features = ['Temperature', 'Pressure', 'Humidity', 'WindDirection', 'Speed']
    return df[features]

print("Preprocessing data...")
X = preprocess_data(data)
y = data['Radiation']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training model...")
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\nEvaluating model...")
y_pred = model.predict(X_test)
print(f'R² Score: {r2_score(y_test, y_pred):.4f}')
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}')
print(f'Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')

print("\nFeature Importance:")
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance)

print("\nSaving model...")
joblib.dump(model, 'model_predict.joblib')
print("Model saved as 'model_predict.joblib'")