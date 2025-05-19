# Install libraries jika dibutuhkan (aktifkan saat di Colab)
# !pip install folium

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import folium

# Load data
url = '/content/covid_19_indonesia_time_series_all.csv'  # Ganti path jika di Colab
df = pd.read_csv(url)

# ======================
# 1. Preprocessing Data
# ======================
df_clean = df.copy()
df_clean = df_clean[df_clean['Location Level'] == 'Province']
df_clean = df_clean.dropna(subset=[
    'Total Cases', 'Total Deaths', 'Total Recovered',
    'Population Density', 'Case Fatality Rate'
])
df_clean['Case Fatality Rate'] = df_clean['Case Fatality Rate'].str.replace('%', '').astype(float)

# Ambil data terbaru per lokasi
latest_df = df_clean.sort_values('Date').groupby('Location').tail(1)

# ==========================================
# 2. Supervised Learning - Linear Regression
# ==========================================
features = ['Total Deaths', 'Total Recovered', 'Population Density', 'Case Fatality Rate']
target = 'Total Cases'

X = latest_df[features]
y = latest_df[target]

model = LinearRegression()
model.fit(X, y)

latest_df['Predicted Cases'] = model.predict(X)

# ==========================================
# 3. Clustering dengan KMeans
# ==========================================
cluster_features = ['Total Cases', 'Total Deaths', 'Total Recovered', 'Population Density']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(latest_df[cluster_features])

kmeans = KMeans(n_clusters=4, random_state=42)
latest_df['Cluster'] = kmeans.fit_predict(X_scaled)

# ==========================================
# 4. Visualisasi Clustering
# ==========================================
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=latest_df, x='Population Density', y='Total Cases',
    hue='Cluster', palette='Set2'
)
plt.title('Clustering Provinsi berdasarkan Total Kasus dan Kepadatan Penduduk')
plt.xlabel('Population Density')
plt.ylabel('Total Cases')
plt.grid(True)
plt.tight_layout()
plt.savefig('clustering_plot.png')
plt.show()

# ==========================================
# 5. Peta Interaktif dengan Folium
# ==========================================
map_cluster = folium.Map(location=[-2.5, 118], zoom_start=5)
colors = ['red', 'blue', 'green', 'purple']

for _, row in latest_df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=8,
        color=colors[row['Cluster']],
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['Location']} (Cluster {row['Cluster']})"
    ).add_to(map_cluster)

# Simpan ke file
map_cluster.save('map_clusters.html')
latest_df.to_csv('processed_covid_data.csv', index=False)
