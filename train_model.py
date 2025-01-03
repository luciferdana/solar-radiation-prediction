import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set style untuk visualisasi yang lebih baik
plt.style.use('seaborn-v0_8-darkgrid')  # menggunakan style yang valid
sns.set_theme(style="darkgrid")

# Load data
print("Loading data...")
data = pd.read_csv('SolarPrediction.csv')

# Data preprocessing
def preprocess_data(df):
    df = df.copy()
    
    # Drop unnecessary columns
    if 'Data' in df.columns:
        df.drop(columns=['Data'], inplace=True)
    
    # Convert time columns
    df['Time'] = pd.to_datetime(df['Time'])
    df['TimeSunRise'] = pd.to_datetime(df['TimeSunRise'])
    df['TimeSunSet'] = pd.to_datetime(df['TimeSunSet'])
    
    # Extract time features
    df['Time_Hour'] = df['Time'].dt.hour
    df['Time_Minute'] = df['Time'].dt.minute
    df['Time_Second'] = df['Time'].dt.second
    
    df['SunriseHour'] = df['TimeSunRise'].dt.hour
    df['SunriseMinute'] = df['TimeSunRise'].dt.minute
    
    df['SunsetHour'] = df['TimeSunSet'].dt.hour
    df['SunsetMinute'] = df['TimeSunSet'].dt.minute
    
    # Calculate day length and time since sunrise
    df['DayLength'] = ((df['SunsetHour'] * 60 + df['SunsetMinute']) - 
                      (df['SunriseHour'] * 60 + df['SunriseMinute']))
    
    df['MinutesSinceSunrise'] = ((df['Time_Hour'] * 60 + df['Time_Minute']) - 
                                (df['SunriseHour'] * 60 + df['SunriseMinute']))
    
    # Drop original time columns
    df.drop(['Time', 'TimeSunRise', 'TimeSunSet'], axis=1, inplace=True)
    
    # Convert all numeric columns to float64
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].astype('float64')
    
    return df

# Function to create visualizations
def create_model_visualizations(X_test, y_test, y_pred, model, feature_names):
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Actual vs Predicted Plot
    ax1 = plt.subplot(2, 2, 1)
    scatter = ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Radiation')
    ax1.set_ylabel('Predicted Radiation')
    ax1.set_title('Actual vs Predicted Radiation')
    
    # 2. Feature Importance Plot
    importance = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.barh(feature_importance['feature'], feature_importance['importance'])
    ax2.set_xlabel('Importance')
    ax2.set_title('Feature Importance')
    
    # 3. Residual Plot
    residuals = y_test - y_pred
    ax3 = plt.subplot(2, 2, 3)
    ax3.scatter(y_pred, residuals, alpha=0.5)
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals')
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_title('Residual Plot')
    
    # 4. Error Distribution Plot
    ax4 = plt.subplot(2, 2, 4)
    sns.histplot(data=residuals, kde=True, ax=ax4)
    ax4.set_xlabel('Prediction Error')
    ax4.set_ylabel('Count')
    ax4.set_title('Error Distribution')
    
    plt.tight_layout()
    plt.show()

    # Print detailed metrics
    print('\nDetailed Model Performance Metrics:')
    print(f'R² Score: {r2_score(y_test, y_pred):.4f}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}')
    print(f'Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}')
    
    # Calculate and print feature importance details
    print('\nFeature Importance Rankings:')
    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")

# Preprocess data
print("Preprocessing data...")
data = preprocess_data(data)

# Split features and target
X = data.drop('Radiation', axis=1)
y = data['Radiation']

# Train test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Create visualizations
print("\nCreating visualizations...")
create_model_visualizations(X_test, y_test, y_pred, model, X.columns)

# Save model
print("\nSaving model...")
try:
    joblib.dump(model, 'model_solar.joblib', compress=3)
    print("Model saved successfully as 'model_solar.joblib'")
except Exception as e:
    print(f"Error saving model: {str(e)}")

# Verify model was saved
if os.path.exists('model_solar.joblib'):
    print(f"Model file size: {os.path.getsize('model_solar.joblib') / 1024 / 1024:.2f} MB")
else:
    print("Model file not found!")