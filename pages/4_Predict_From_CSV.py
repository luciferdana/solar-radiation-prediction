import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.title('Prediksi Radiasi Matahari dari CSV')

@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_predict.joblib')
        st.sidebar.success("✅ Model berhasil dimuat")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Gagal memuat model: {str(e)}")
        return None


def preprocess_csv(df):
    try:
        df = df.copy()
        
        column_mapping = {
            'humadity': 'Humidity',
            'temperature': 'Temperature',
            'pressure': 'Pressure',
            'wind_direction': 'WindDirection',
            'speed': 'Speed',
            'radiation': 'Radiation'  # Tambahan kolom radiasi
        }
        df.rename(columns=column_mapping, inplace=True)
        
        for col in df.columns:
            if col not in ['sunrise', 'sunset']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        features = ['Temperature', 'Pressure', 'Humidity', 'WindDirection', 'Speed']
        missing_columns = [col for col in features if col not in df.columns]
        if missing_columns:
            st.error(f"Kolom yang tidak ditemukan: {', '.join(missing_columns)}")
            return None
            
        actual_radiation = None
        if 'Radiation' in df.columns:
            actual_radiation = df['Radiation'].values
            
        return df[features], actual_radiation
    
    except Exception as e:
        st.error(f"Error dalam preprocessing: {str(e)}")
        return None

model = load_model()


if model is None:
    st.error("Model tidak dapat dimuat. Periksa file model.")
else:
    uploaded_file = st.file_uploader("Upload file CSV untuk prediksi", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success(f"Berhasil membaca {len(data)} baris data")
            
            st.write("Preview Data Original:")
            st.write(data.head())
            
            processed_data, actual_radiation = preprocess_csv(data)
            
            if processed_data is not None:
                st.write("\nPreview Data Setelah Preprocessing:")
                st.write(processed_data.head())
                
                try:
                    predictions = model.predict(processed_data)
                    
                    results = data.copy()
                    results['Predicted_Radiation'] = predictions
                    
                    def categorize(value):
                        if value <= 200: return 'Rendah'
                        elif value <= 500: return 'Sedang'
                        elif value <= 800: return 'Tinggi'
                        else: return 'Sangat Tinggi'
                    
                    results['Category'] = results['Predicted_Radiation'].apply(categorize)
                    
                    st.write("\nHasil Prediksi:")
                    st.write(results.head())
                    
                    st.write("\nStatistik Prediksi:")
                    stats = pd.DataFrame({
                        'Metrik': ['Rata-rata', 'Minimum', 'Maksimum'],
                        'Nilai (W/m²)': [
                            f"{np.mean(predictions):.2f}",
                            f"{np.min(predictions):.2f}",
                            f"{np.max(predictions):.2f}"
                        ]
                    })
                    st.table(stats)

                    # Visualisasi distribusi prediksi
                    st.write("\nDistribusi Nilai Prediksi:")
                    
                    # Histogram prediksi
                    fig_pred = px.histogram(
                        predictions,
                        title='Distribusi Nilai Prediksi Radiasi',
                        labels={'value': 'Radiasi (W/m²)', 'count': 'Frekuensi'},
                        color_discrete_sequence=['indianred']
                    )
                    st.plotly_chart(fig_pred)
                    
                    # Line plot trend prediksi
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        y=predictions,
                        name='Prediksi Radiasi',
                        line=dict(color='red')
                    ))
                    fig_trend.update_layout(
                        title='Trend Prediksi Radiasi',
                        xaxis_title='Index Data',
                        yaxis_title='Radiasi (W/m²)',
                        height=500
                    )
                    st.plotly_chart(fig_trend)
                    
                    # Menampilkan metrik akurasi jika ada data aktual
                    if actual_radiation is not None:
                        st.write("\nMetrik Akurasi Model:")
                        metrics = calculate_metrics(actual_radiation, predictions)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("R² Score", f"{metrics['R² Score']:.4f}")
                            st.metric("Mean Absolute Error", f"{metrics['Mean Absolute Error']:.4f}")
                        with col2:
                            st.metric("Root Mean Squared Error", f"{metrics['Root Mean Squared Error']:.4f}")
                            st.metric("Mean Squared Error", f"{metrics['Mean Squared Error']:.4f}")
                        
                        # Plot perbandingan aktual vs prediksi
                        fig_compare = go.Figure()
                        fig_compare.add_trace(go.Scatter(
                            y=actual_radiation,
                            name='Aktual',
                            line=dict(color='blue')
                        ))
                        fig_compare.add_trace(go.Scatter(
                            y=predictions,
                            name='Prediksi',
                            line=dict(color='red')
                        ))
                        fig_compare.update_layout(
                            title='Perbandingan Nilai Aktual vs Prediksi',
                            xaxis_title='Index Data',
                            yaxis_title='Radiasi (W/m²)',
                            height=500
                        )
                        st.plotly_chart(fig_compare)
                    
                    csv = results.to_csv(index=False)
                    st.download_button(
                        "Download Hasil Prediksi",
                        csv,
                        "hasil_prediksi.csv",
                        "text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Error dalam prediksi: {str(e)}")
        
        except Exception as e:
            st.error(f"Error membaca file CSV: {str(e)}")