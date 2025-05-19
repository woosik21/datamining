import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# ------------------------------
# DATA LOADING & CLEANING
# ------------------------------
st.title("📊 Dashboard & Analisis COVID-19 Indonesia")

df = pd.read_csv('covid_19_indonesia_time_series_all.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df[df['Location Level'] == 'Province']
latest_df = df.sort_values('Date').groupby('Location').tail(1)

latest_df['Case Fatality Rate'] = latest_df['Case Fatality Rate'].str.replace('%', '').astype(float)
data = latest_df[[
    'Location', 'Latitude', 'Longitude',
    'Total Cases', 'Total Deaths', 'Total Recovered',
    'Population', 'Population Density', 'Case Fatality Rate'
]].dropna()

# ------------------------------
# SUPERVISED LEARNING - REGRESSION
# ------------------------------
st.header("🧠 Prediksi Total Kasus COVID-19 (Regresi)")
X = data[['Total Deaths', 'Total Recovered', 'Population', 'Population Density', 'Case Fatality Rate']]
y = data['Total Cases']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
score = reg.score(X_test, y_test)

st.write("**R² Score (Akurasi Test):**", round(score, 2))
st.write("**Koefisien:**", dict(zip(X.columns, reg.coef_.round(2))))

# ------------------------------
# UNSUPERVISED LEARNING - KMEANS
# ------------------------------
st.header("🔍 Clustering Wilayah (KMeans)")
features = data[['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']]
kmeans = KMeans(n_clusters=3, random_state=0)
data['Cluster'] = kmeans.fit_predict(features)

# ------------------------------
# INTERACTIVE MAP
# ------------------------------
st.subheader("🗺️ Peta Interaktif Klaster COVID-19")
fig_map = px.scatter_mapbox(
    data,
    lat='Latitude', lon='Longitude',
    color='Cluster',
    hover_name='Location',
    size='Total Cases',
    zoom=4,
    mapbox_style='open-street-map'
)
st.plotly_chart(fig_map)

# ------------------------------
# DAILY CASE TREND CHART
# ------------------------------
st.subheader("📈 Tren Kasus Harian Nasional")
indo_df = df[df['Location'] == 'Indonesia']
st.line_chart(indo_df.set_index('Date')['New Cases'])

# ------------------------------
# RISK SUMMARY
# ------------------------------
st.subheader("📌 Ringkasan Risiko Wilayah per Cluster")
for i in range(3):
    wilayah = data[data['Cluster'] == i]['Location'].tolist()
    st.markdown(f"**Cluster {i}** - Jumlah Wilayah: {len(wilayah)}")
    st.write(wilayah)
