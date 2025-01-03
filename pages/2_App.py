import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import pickle
import os
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Prediksi Radiasi Matahari",
    page_icon="☀️",
    layout="wide"
)

st.title('Aplikasi Prediksi Radiasi Matahari')

# Tab untuk memilih antara Prediksi dan Analisis Akurasi
tab1, tab2 = st.tabs(["Prediksi", "Analisis Akurasi"])

with tab1:
    # Initialize session state for prediction history
    if 'input_history' not in st.session_state:
        st.session_state.input_history = pd.DataFrame(columns=[
            'UNIXTime', 'Time', 'Temperature', 'Pressure', 'Humidity',
            'WindDirection', 'Speed', 'Sunrise', 'Sunset', 'Prediction'
        ])

    @st.cache_resource
    def load_model():
        try:
            model = joblib.load('model_solar.joblib')
            return model
        except Exception as e:
            try:
                with open('model_solar.pkl', 'rb') as f:
                    model = pickle.load(f)
                return model
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                return None

    def create_trend_visualization():
        if len(st.session_state.input_history) == 0:
            return
        
        try:
            df = st.session_state.input_history.copy()
            
            fig = make_subplots(
                rows=3, cols=3,
                subplot_titles=(
                    'Temperature over Time', 
                    'Pressure over Time',
                    'Humidity over Time', 
                    'Wind Direction over Time',
                    'Wind Speed over Time', 
                    'Sunrise Time over Time',
                    'Sunset Time over Time', 
                    'UNIX Time over Time',
                    'Radiation Prediction over Time'
                ),
                vertical_spacing=0.2,
                horizontal_spacing=0.1
            )

            x_points = list(range(1, len(df) + 1))

            plots = [
                {'var': 'Temperature', 'title': 'Temperature (°C)', 'row': 1, 'col': 1, 'color': 'red'},
                {'var': 'Pressure', 'title': 'Pressure (hPa)', 'row': 1, 'col': 2, 'color': 'blue'},
                {'var': 'Humidity', 'title': 'Humidity (%)', 'row': 1, 'col': 3, 'color': 'green'},
                {'var': 'WindDirection', 'title': 'Wind Direction (°)', 'row': 2, 'col': 1, 'color': 'purple'},
                {'var': 'Speed', 'title': 'Wind Speed (m/s)', 'row': 2, 'col': 2, 'color': 'orange'},
                {'var': 'Sunrise', 'title': 'Sunrise Time', 'row': 2, 'col': 3, 'color': 'cyan'},
                {'var': 'Sunset', 'title': 'Sunset Time', 'row': 3, 'col': 1, 'color': 'magenta'},
                {'var': 'UNIXTime', 'title': 'UNIX Time', 'row': 3, 'col': 2, 'color': 'yellow'},
                {'var': 'Prediction', 'title': 'Radiation (W/m²)', 'row': 3, 'col': 3, 'color': 'brown'}
            ]

            for plot in plots:
                fig.add_trace(
                    go.Scatter(
                        x=x_points,
                        y=df[plot['var']].values,
                        name=plot['title'],
                        mode='lines+markers',
                        line=dict(color=plot['color'], width=2),
                        marker=dict(size=8)
                    ),
                    row=plot['row'],
                    col=plot['col']
                )

                fig.update_xaxes(
                    title_text='Measurement Number',
                    row=plot['row'],
                    col=plot['col']
                )
                fig.update_yaxes(
                    title_text=plot['title'],
                    row=plot['row'],
                    col=plot['col']
                )

            fig.update_layout(
                height=1200,
                showlegend=False,
                title_text="Analysis of Solar Radiation Parameters",
                title_x=0.5,
                template="plotly_dark"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Parameter Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Temperature", f"{df['Temperature'].mean():.2f}°C")
                st.metric("Average Pressure", f"{df['Pressure'].mean():.2f} hPa")
                st.metric("Average Humidity", f"{df['Humidity'].mean():.2f}%")
            
            with col2:
                st.metric("Average Wind Speed", f"{df['Speed'].mean():.2f} m/s")
                st.metric("Average Wind Direction", f"{df['WindDirection'].mean():.2f}°")
                st.metric("Total Measurements", len(df))
            
            with col3:
                st.metric("Average Radiation", f"{df['Prediction'].mean():.2f} W/m²")
                st.metric("Max Radiation", f"{df['Prediction'].max():.2f} W/m²")
                st.metric("Min Radiation", f"{df['Prediction'].min():.2f} W/m²")

        except Exception as e:
            st.error(f"Visualization error: {str(e)}")

    def preprocess_data(unix_time, time, temperature, pressure, humidity, wind_direction, 
                       speed, sunrise, sunset):
        try:
            time_dt = datetime.strptime(time, '%H:%M:%S')
            sunrise_dt = datetime.strptime(sunrise, '%H:%M')
            sunset_dt = datetime.strptime(sunset, '%H:%M')
            
            time_hour = float(time_dt.hour)
            time_minute = float(time_dt.minute)
            time_second = float(time_dt.second)
            
            sunrise_hour = float(sunrise_dt.hour)
            sunrise_minute = float(sunrise_dt.minute)
            
            sunset_hour = float(sunset_dt.hour)
            sunset_minute = float(sunset_dt.minute)
            
            day_length = float((sunset_hour * 60 + sunset_minute) - 
                             (sunrise_hour * 60 + sunrise_minute))
            
            minutes_since_sunrise = float((time_hour * 60 + time_minute) - 
                                       (sunrise_hour * 60 + sunrise_minute))
            
            data = pd.DataFrame({
                'UNIXTime': [float(unix_time)],
                'Temperature': [float(temperature)],
                'Pressure': [float(pressure)],
                'Humidity': [float(humidity)],
                'WindDirection(Degrees)': [float(wind_direction)],
                'Speed': [float(speed)],
                'Time_Hour': [time_hour],
                'Time_Minute': [time_minute],
                'Time_Second': [time_second],
                'SunriseHour': [sunrise_hour],
                'SunriseMinute': [sunrise_minute],
                'SunsetHour': [sunset_hour],
                'SunsetMinute': [sunset_minute],
                'DayLength': [day_length],
                'MinutesSinceSunrise': [minutes_since_sunrise]
            })
            
            for col in data.columns:
                data[col] = data[col].astype('float64')
            
            return data
        except Exception as e:
            st.error(f"Error in preprocessing data: {str(e)}")
            return None

    # Load model
    model = load_model()

    if model is None:
        st.error("Model could not be loaded. Please ensure model file is available.")
    else:
        # Input form
        with st.form("solar_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                unix_time = st.number_input("UNIX Time:", min_value=0, value=1609459200)
                time = st.text_input("Time (HH:MM:SS):", "12:00:00")
                temperature = st.number_input("Temperature (°C):", value=25.0, format="%.2f")
                pressure = st.number_input("Pressure (hPa):", value=1013.0, format="%.2f")
                humidity = st.number_input("Humidity (%):", min_value=0.0, max_value=100.0, value=50.0, format="%.2f")
            
            with col2:
                wind_direction = st.number_input("Wind Direction (°):", min_value=0.0, max_value=360.0, value=180.0, format="%.2f")
                speed = st.number_input("Wind Speed (m/s):", min_value=0.0, value=5.0, format="%.2f")
                sunrise = st.text_input("Sunrise Time (HH:MM):", "06:00")
                sunset = st.text_input("Sunset Time (HH:MM):", "18:00")

            submit = st.form_submit_button("Predict")

        if submit:
            try:
                input_data = preprocess_data(
                    unix_time, time, temperature, pressure, humidity,
                    wind_direction, speed, sunrise, sunset
                )
                
                if input_data is not None:
                    prediction = model.predict(input_data)[0]
                    
                    new_data = pd.DataFrame({
                        'UNIXTime': [unix_time],
                        'Time': [time],
                        'Temperature': [temperature],
                        'Pressure': [pressure],
                        'Humidity': [humidity],
                        'WindDirection': [wind_direction],
                        'Speed': [speed],
                        'Sunrise': [sunrise],
                        'Sunset': [sunset],
                        'Prediction': [prediction]
                    })
                    
                    st.session_state.input_history = pd.concat([
                        st.session_state.input_history, 
                        new_data
                    ], ignore_index=True)

                    st.success(f"Solar Radiation Prediction: {prediction:.2f} W/m²")
                    
                    if prediction <= 200:
                        category = "Low"
                        color = "red"
                    elif prediction <= 500:
                        category = "Medium"
                        color = "orange"
                    elif prediction <= 800:
                        category = "High"
                        color = "blue"
                    else:
                        category = "Very High"
                        color = "green"
                        
                    st.markdown(
                        f"Category: <span style='color:{color}'>{category}</span>", 
                        unsafe_allow_html=True
                    )
                    
                    create_trend_visualization()
                    
                    with st.expander("View History Data"):
                        st.dataframe(st.session_state.input_history)
                    
            except Exception as e:
                st.error(f"Processing error: {str(e)}")

with tab2:
    st.header("Model Accuracy Analysis")
    
    # Load training data
    try:
        data = pd.read_csv('SolarPrediction.csv')
        
        # Preprocess data for analysis
        def preprocess_data_analysis(df):
            df = df.copy()
            if 'Data' in df.columns:
                df.drop(columns=['Data'], inplace=True)
            
            df['Time'] = pd.to_datetime(df['Time'])
            df['TimeSunRise'] = pd.to_datetime(df['TimeSunRise'])
            df['TimeSunSet'] = pd.to_datetime(df['TimeSunSet'])
            
            # Extract features
            df['Time_Hour'] = df['Time'].dt.hour
            df['Time_Minute'] = df['Time'].dt.minute
            df['Time_Second'] = df['Time'].dt.second
            df['SunriseHour'] = df['TimeSunRise'].dt.hour
            df['SunriseMinute'] = df['TimeSunRise'].dt.minute
            df['SunsetHour'] = df['TimeSunSet'].dt.hour
            df['SunsetMinute'] = df['TimeSunSet'].dt.minute
            
            df['DayLength'] = ((df['SunsetHour'] * 60 + df['SunsetMinute']) - 
                             (df['SunriseHour'] * 60 + df['SunriseMinute']))
            df['MinutesSinceSunrise'] = ((df['Time_Hour'] * 60 + df['Time_Minute']) - 
                                       (df['SunriseHour'] * 60 + df['SunriseMinute']))
            
            df.drop(['Time', 'TimeSunRise', 'TimeSunSet'], axis=1, inplace=True)
            
            return df

        processed_data = preprocess_data_analysis(data)
        X = processed_data.drop('Radiation', axis=1)
        y = processed_data['Radiation']
        
        # Make predictions on entire dataset
        predictions = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, predictions)
        mae = mean_absolute_error(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{r2:.4f}")
        with col2:
            st.metric("Mean Absolute Error", f"{mae:.4f}")
        with col3:
            st.metric("Root Mean Squared Error", f"{rmse:.4f}")
        
        # Create accuracy visualizations
        st.subheader("Model Performance Visualizations")
        
        # Actual vs Predicted Plot
        fig_scatter = go.Figure()
        fig_scatter.add_trace(
            go.Scatter(
                x=y,
                y=predictions,
                mode='markers',
                marker=dict(size=5, opacity=0.5),
                name='Predictions'
            )
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[y.min(), y.max()],
                y=[y.min(), y.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            )
        )
        fig_scatter.update_layout(
            title='Actual vs Predicted Values',
            xaxis_title='Actual Radiation',
            yaxis_title='Predicted Radiation',
            template='plotly_dark',
            height=600
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Feature Importance Plot
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True)

        fig_importance = go.Figure()
        fig_importance.add_trace(
            go.Bar(
                y=feature_importance['feature'],
                x=feature_importance['importance'],
                orientation='h'
            )
        )
        fig_importance.update_layout(
            title='Feature Importance',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            template='plotly_dark',
            height=800
        )
        st.plotly_chart(fig_importance, use_container_width=True)

        # Residual Plot
        residuals = y - predictions
        fig_residuals = go.Figure()
        fig_residuals.add_trace(
            go.Scatter(
                x=predictions,
                y=residuals,
                mode='markers',
                marker=dict(size=5, opacity=0.5)
            )
        )
        fig_residuals.add_hline(
            y=0,
            line=dict(color='red', dash='dash')
        )
        fig_residuals.update_layout(
            title='Residual Plot',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            template='plotly_dark',
            height=600
        )
        st.plotly_chart(fig_residuals, use_container_width=True)

        # Error Distribution
        fig_error = go.Figure()
        fig_error.add_trace(
            go.Histogram(
                x=residuals,
                nbinsx=50,
                name='Error Distribution'
            )
        )
        fig_error.update_layout(
            title='Error Distribution',
            xaxis_title='Prediction Error',
            yaxis_title='Count',
            template='plotly_dark',
            height=600
        )
        st.plotly_chart(fig_error, use_container_width=True)

        # Display feature importance details
        st.subheader("Feature Importance Rankings")
        feature_importance_sorted = feature_importance.sort_values('importance', ascending=False)
        for idx, row in feature_importance_sorted.iterrows():
            st.write(f"{row['feature']}: {row['importance']:.4f}")

    except Exception as e:
        st.error(f"Error in accuracy analysis: {str(e)}")
