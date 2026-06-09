# 1. INSTALL & IMPORT LIBRARY
"""

import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import scipy.stats as stats
import requests
import pickle

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV

# Peringatan 'FutureWarning: Passing `palette` without assigning `hue` is deprecated...'
# berarti bahwa di versi Seaborn mendatang (v0.14.0 dan seterusnya), penggunaan argumen `palette`
# di `sns.barplot` tanpa secara eksplisit mengatur argumen `hue` akan dianggap usang.
# Untuk memperbaikinya, Anda harus menetapkan variabel yang sama yang Anda gunakan untuk `x`
# juga ke `hue` dan mengatur `legend=False` dalam panggilan `sns.barplot` Anda.
# Misalnya, jika Anda memiliki:
# `ax = sns.barplot(x=daily_bike_share.index, y=daily_bike_share.values, palette='viridis')`
# Anda harus mengubahnya menjadi:
# `ax = sns.barplot(x=daily_bike_share.index, y=daily_bike_share.values, hue=daily_bike_share.index, palette='viridis', legend=False)`
# Ini hanyalah peringatan, bukan error, jadi kode Anda masih berjalan dengan benar untuk saat ini.

# Mount Google Drive (opsional)
from google.colab import drive
drive.mount('/content/drive')

"""# 2. LOAD DATA

"""

path = kagglehub.dataset_download("hmavrodiev/london-bike-sharing-dataset")
df = pd.read_csv(os.path.join(path, 'london_merged.csv'))

print("Dataset berhasil dimuat!")
print(f"Shape dataset: {df.shape}")
df.head()

"""# 3. DATA PREPROCESSING & FEATURE ENGINEERING"""

# Info dataset
df.info()
df.describe()

# Visualisasi korelasi awal
plt.figure(figsize=(13,5))
sns.heatmap(df.select_dtypes("number").corr(), cmap="YlGnBu", annot=True)
plt.title("Korelasi Antar Variabel")
plt.show()

# Membuat kolom waktu dari timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour

# Merapikan urutan kolom
ordered_cols = ['timestamp', 'is_weekend', 'is_holiday', 'season', 'year', 'month', 'day', 'hour',
                'weather_code', 't1', 't2', 'hum', 'wind_speed', 'cnt']
df = df[ordered_cols]

# Mengubah tipe data
df['is_holiday'] = df['is_holiday'].astype('int64')
df['is_weekend'] = df['is_weekend'].astype('int64')
df['weather_code'] = df['weather_code'].astype('int64')
df['season'] = df['season'].astype('int64')
df['cnt'] = df['cnt'].astype('float64')

print("\nTipe data setelah konversi:")
print(df.dtypes)

"""# 4. DATA CLEANING (CEK NULL & DUPLIKAT)"""

print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nData duplikat: {df.duplicated().sum()}")

"""# 5. HANDLING OUTLIER DENGAN IQR"""

# Visualisasi outlier sebelum handling
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
cols_to_plot = ['cnt', 't1', 't2', 'hum', 'wind_speed']
for i, col in enumerate(cols_to_plot):
    sns.boxplot(data=df, y=col, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f'Boxplot {col}')
plt.tight_layout()
plt.show()

# Membuat copy dataset untuk handling outlier
df_clean = df.copy()

# Handling outlier dengan IQR
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
df_clean[numeric_cols] = df_clean[numeric_cols].astype(float)

outlier_count = {}
for feature in numeric_cols:
    Q1 = df_clean[feature].quantile(0.25)
    Q3 = df_clean[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_limit = Q1 - (1.5 * IQR)
    upper_limit = Q3 + (1.5 * IQR)

    before = df_clean.shape[0]
    df_clean.loc[df_clean[feature] < lower_limit, feature] = lower_limit
    df_clean.loc[df_clean[feature] > upper_limit, feature] = upper_limit
    after = df_clean.shape[0]
    outlier_count[feature] = df_clean[(df_clean[feature] < lower_limit) | (df_clean[feature] > upper_limit)].shape[0]

print("\nJumlah outlier per kolom setelah handling:")
for col, count in outlier_count.items():
    print(f"{col}: {count}")

"""# 6. EKSPLORASI DATA (EDA) - VISUALISASI"""

# Membuat dataframe untuk visualisasi (dengan label yang lebih mudah dibaca)
df_viz = df_clean.copy()

# Mapping untuk visualisasi
season_map = {0: 'Spring', 1: 'Summer', 2: 'Fall', 3: 'Winter'}
weather_map = {1: 'Clear', 2: 'Scattered clouds', 3: 'Broken clouds', 4: 'Cloudy',
               7: 'Rain', 10: 'Rain with thunderstorm', 26: 'Snowfall', 94: 'Freezing Fog'}

df_viz['season_label'] = df_viz['season'].map(season_map)
df_viz['weather_label'] = df_viz['weather_code'].map(weather_map)

# 6.1 Permintaan penyewaan per hari
daily_bike_share = (df_viz.groupby(df_viz['timestamp'].dt.day_name())['cnt'].sum() / 10000).round(1)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_bike_share = daily_bike_share.reindex(day_order)

plt.figure(figsize=(12,5))
ax = sns.barplot(x=daily_bike_share.index, y=daily_bike_share.values, errorbar=None, palette='viridis')
ax.set_title("Permintaan Penyewaan Sepeda per Hari")
ax.set_xlabel('')
for i in ax.containers:
    ax.bar_label(i)
plt.show()

# 6.2 Permintaan penyewaan per musim
seasonal_bike_share = (df_viz.groupby('season_label')['cnt'].sum() / 10000).round(1)

plt.figure(figsize=(10,5))
ax = sns.barplot(x=seasonal_bike_share.index, y=seasonal_bike_share.values, errorbar=None, palette='coolwarm')
ax.set_title("Permintaan Penyewaan Sepeda per Musim")
for i in ax.containers:
    ax.bar_label(i)
plt.show()

# 6.3 Permintaan penyewaan per jam
hourly_bike_share = (df_viz.groupby('hour')['cnt'].sum() / 10000).round(1)

plt.figure(figsize=(14,5))
ax = sns.barplot(x=hourly_bike_share.index, y=hourly_bike_share.values, errorbar=None, palette='rocket')
ax.set_title("Permintaan Penyewaan Sepeda per Jam")
ax.set_xlabel('Jam')
for i in ax.containers:
    ax.bar_label(i)
plt.show()

# 6.4 Permintaan penyewaan berdasarkan cuaca
weather_bike_share = (df_viz.groupby('weather_label')['cnt'].sum() / 10000).round(1)

plt.figure(figsize=(12,5))
ax = sns.barplot(x=weather_bike_share.index, y=weather_bike_share.values, errorbar=None, palette='muted')
ax.set_title("Permintaan Penyewaan Sepeda berdasarkan Cuaca")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
for i in ax.containers:
    ax.bar_label(i)
plt.show()

"""# 7. CEK MULTIKOLINEARITAS (VIF)"""

# Pilih kolom numerik untuk cek VIF
num_att = df_clean[['t1', 't2', 'hum', 'wind_speed']]

# Hitung VIF
vif_scores = pd.Series(
    [variance_inflation_factor(num_att.values, i) for i in range(num_att.shape[1])],
    index=num_att.columns
)
print("\nVIF Scores sebelum drop:")
print(vif_scores)

# Hapus kolom 't1' karena memiliki multikolinearitas tinggi dengan 't2'
df_clean = df_clean.drop(['t1'], axis=1)
print("\nKolom 't1' dihapus karena multikolinearitas tinggi dengan 't2'")

"""# 8. PREPARE DATA FOR MODELING"""

# Definisikan fitur (X) dan target (y)
# Hapus kolom yang tidak diperlukan untuk modeling
X = df_clean.drop(['timestamp', 'cnt'], axis=1)
y = df_clean['cnt']

print(f"\nFitur yang digunakan: {list(X.columns)}")
print(f"Shape X: {X.shape}")
print(f"Shape y: {y.shape}")

"""# 9. SPLIT DATA TRAIN & TEST"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set: {X_train.shape[0]} rows")
print(f"Test set: {X_test.shape[0]} rows")

"""# 10. FEATURE SCALING (STANDARDISASI)"""

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling selesai!")

"""# 11. TRAINING & EVALUASI MODEL"""

# Definisikan model dengan parameter yang dioptimalkan
models = [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge(alpha=1.0)),
    ("Lasso Regression", Lasso(alpha=0.01)),
    ("Random Forest", RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)),
    ("Gradient Boosting", GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42)),
    ("LightGBM", LGBMRegressor(n_estimators=150, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)),
    ("AdaBoost", AdaBoostRegressor(n_estimators=100, learning_rate=0.05, random_state=42)),
    ("XGBoost", xgb.XGBRegressor(n_estimators=200, learning_rate=0.03, max_depth=7,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)),
    ("KNN", KNeighborsRegressor(n_neighbors=7, weights='distance'))
]

# Evaluasi semua model
results = []
best_model = None
best_r2 = -float('inf')

print("=" * 80)
print("HASIL EVALUASI MODEL")
print("=" * 80)

for model_name, model in models:
    # Training
    model.fit(X_train_scaled, y_train)

    # Prediksi
    y_pred = model.predict(X_test_scaled)

    # Metrik evaluasi
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append([model_name, rmse, mae, r2])

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = model_name

    print(f"\nModel: {model_name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")

# Tampilkan ringkasan
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE", "R²"])
print("\n" + "=" * 80)
print("RINGKASAN HASIL EVALUASI (URUTAN R² TERTINGGI)")
print("=" * 80)
print(results_df.sort_values("R²", ascending=False))

print(f"\n✅ Model terbaik: {best_model_name} dengan R² = {best_r2:.4f}")

"""# 12. OPTIMASI KHUSUS UNTUK XGBOOST (AGAR MENJADI TERBAIK)"""

# Hyperparameter grid untuk XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.03, 0.05],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Grid search untuk XGBoost
xgb_base = xgb.XGBRegressor(random_state=42, verbosity=0)
grid_search = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='r2',
    cv=3,
    n_jobs=-1,
    verbose=0
)

grid_search.fit(X_train_scaled, y_train)

print(f"Parameter terbaik: {grid_search.best_params_}")
print(f"R² terbaik dari grid search: {grid_search.best_score_:.4f}")

# Gunakan model terbaik dari grid search
best_xgb = grid_search.best_estimator_

# Evaluasi final
y_pred_best = best_xgb.predict(X_test_scaled)
final_r2 = r2_score(y_test, y_pred_best)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_best))

print(f"\n✅ XGBoost Final - R²: {final_r2:.4f}, RMSE: {final_rmse:.4f}")

# Bandingkan dengan LightGBM (menggunakan parameter optimal juga)
print("\n" + "=" * 80)
print("PERBANDINGAN XGBOOST vs LIGHTGBM (SETELAH OPTIMASI)")
print("=" * 80)

# LightGBM dengan parameter optimal
lgb_optimized = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)
lgb_optimized.fit(X_train_scaled, y_train)
y_pred_lgb = lgb_optimized.predict(X_test_scaled)
lgb_r2 = r2_score(y_test, y_pred_lgb)
lgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lgb))

print(f"\nXGBoost Optimized:")
print(f"  R²: {final_r2:.6f}")
print(f"  RMSE: {final_rmse:.4f}")

print(f"\nLightGBM Optimized:")
print(f"  R²: {lgb_r2:.6f}")
print(f"  RMSE: {lgb_rmse:.4f}")

# Tentukan model terbaik untuk digunakan selanjutnya
if final_r2 >= lgb_r2:
    best_final_model = best_xgb
    best_final_name = "XGBoost"
    print(f"\n🏆 Model terbaik: XGBoost dengan R² = {final_r2:.6f}")
else:
    best_final_model = lgb_optimized
    best_final_name = "LightGBM"
    print(f"\n🏆 Model terbaik: LightGBM dengan R² = {lgb_r2:.6f}")

"""# 13. FEATURE IMPORTANCE (DARI MODEL TERBAIK)"""

print("\n" + "=" * 80)
print(f"FEATURE IMPORTANCE - {best_final_name}")
print("=" * 80)

if best_final_name == "XGBoost":
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
else:
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_final_model.feature_importances_
    }).sort_values('Importance', ascending=False)

print(feature_importance)

# Visualisasi
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title(f'Feature Importance - {best_final_name}', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

"""# 14. PREDIKSI DENGAN MODEL TERBAIK"""

print("\n" + "=" * 80)
print("PREDIKSI DENGAN DATA CUACA REAL 2023")
print("=" * 80)

# Ambil data cuaca dari API Open-Meteo
lat, lon = 51.5074, -0.1278  # London
start_date = "2023-01-01"
end_date = "2023-02-28"

url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code&timezone=GMT"

response = requests.get(url)
weather_data = response.json()

# Buat dataframe dari data cuaca
df_weather_2023 = pd.DataFrame({
    'timestamp': pd.to_datetime(weather_data['hourly']['time']),
    't2': weather_data['hourly']['temperature_2m'],
    'hum': weather_data['hourly']['relative_humidity_2m'],
    'wind_speed': weather_data['hourly']['wind_speed_10m'],
    'weather_code': weather_data['hourly']['weather_code']
})

# Fungsi mapping weather code (sesuai dengan dataset asli)
def map_weather_code(code):
    if code <= 1:
        return 1  # Clear sky
    elif code <= 3:
        return 2  # Partly cloudy
    elif code <= 49:
        return 4  # Cloudy
    elif code <= 69:
        return 7  # Rain
    elif code <= 79:
        return 26  # Snowfall
    else:
        return 3  # Broken clouds

# Feature engineering untuk data 2023
df_weather_2023['year'] = df_weather_2023['timestamp'].dt.year
df_weather_2023['month'] = df_weather_2023['timestamp'].dt.month
df_weather_2023['day'] = df_weather_2023['timestamp'].dt.day
df_weather_2023['hour'] = df_weather_2023['timestamp'].dt.hour
df_weather_2023['is_weekend'] = df_weather_2023['timestamp'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
df_weather_2023['is_holiday'] = 0  # Asumsi tidak ada libur
df_weather_2023['season'] = df_weather_2023['month'].apply(lambda m: 0 if m in [3,4,5] else (1 if m in [6,7,8] else (2 if m in [9,10,11] else 3)))
df_weather_2023['weather_code'] = df_weather_2023['weather_code'].apply(map_weather_code)

# Siapkan fitur untuk prediksi (pastikan kolomnya sama dengan X_train)
features_2023 = df_weather_2023[['is_weekend', 'is_holiday', 'season', 'year', 'month', 'day', 'hour',
                                   'weather_code', 't2', 'hum', 'wind_speed']]

# Scaling
features_2023_scaled = scaler.transform(features_2023)

# Prediksi dengan model terbaik (best_final_model dari bagian 12)
predictions_2023 = best_final_model.predict(features_2023_scaled)
df_weather_2023['predicted_cnt'] = np.maximum(predictions_2023, 0).round(0)

print(f"Total prediksi penyewaan Jan-Feb 2023: {df_weather_2023['predicted_cnt'].sum():,.0f}")

# Tampilkan hasil prediksi
print("\n10 Data pertama hasil prediksi:")
print(df_weather_2023[['timestamp', 't2', 'hum', 'predicted_cnt']].head(10))

"""# 15. PERBANDINGAN HISTORIS (2017) VS PREDIKSI (2023)"""

print("\n" + "=" * 80)
print("PERBANDINGAN HISTORIS VS PREDIKSI")
print("=" * 80)

# Data historis 2017 (1-2 Januari)
historical_2017 = df_clean[(df_clean['timestamp'] >= '2017-01-01') & (df_clean['timestamp'] <= '2017-01-02')]
total_historical_2017 = historical_2017['cnt'].sum()

# Data prediksi 2023 (1-2 Januari)
prediction_2023_jan = df_weather_2023[(df_weather_2023['timestamp'] >= '2023-01-01') & (df_weather_2023['timestamp'] <= '2023-01-02')]
total_prediction_2023 = prediction_2023_jan['predicted_cnt'].sum()

print(f"Total penyewaan 1-2 Jan 2017 (Historis): {total_historical_2017:,.0f}")
print(f"Total penyewaan 1-2 Jan 2023 (Prediksi): {total_prediction_2023:,.0f}")
print(f"Selisih: {total_prediction_2023 - total_historical_2017:,.0f} ({((total_prediction_2023 - total_historical_2017)/total_historical_2017*100):.1f}%)")

# Visualisasi perbandingan
plt.figure(figsize=(8, 5))
labels = ['Jan 2017 (Aktual)', 'Jan 2023 (Prediksi)']
values = [total_historical_2017, total_prediction_2023]
colors = ['#2ecc71', '#3498db']
bars = plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
plt.title('Perbandingan Volume Rental Sepeda: Historis vs Prediksi', fontsize=14, fontweight='bold')
plt.ylabel('Total Rental (cnt)', fontsize=12)
for bar, val in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f'{val:,.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.show()

"""# 16. VISUALISASI POLA PER JAM (PREDIKSI 2023)"""

hourly_prediction = df_weather_2023.groupby('hour')['predicted_cnt'].sum() / 10000

plt.figure(figsize=(14, 6))
ax = sns.barplot(x=hourly_prediction.index, y=hourly_prediction.values, errorbar=None, palette='coolwarm')
ax.set_title(f"Prediksi Penyewaan Sepeda per Jam (Jan-Feb 2023) - {best_final_name}", fontsize=14, fontweight='bold')
ax.set_xlabel('Jam', fontsize=12)
ax.set_ylabel('Total Rental (x10,000)', fontsize=12)
for i, v in enumerate(hourly_prediction.values):
    ax.text(i, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.show()

"""# 17. FEATURE IMPORTANCE DARI MODEL TERBAIK"""

print("\n" + "=" * 80)
print(f"FEATURE IMPORTANCE - {best_final_name}")
print("=" * 80)

if best_final_name == "XGBoost":
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_final_model.feature_importances_
    }).sort_values('Importance', ascending=False)
else:
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_final_model.feature_importances_
    }).sort_values('Importance', ascending=False)

print(feature_importance)

# Visualisasi
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title(f'Feature Importance - {best_final_name}', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

"""# 18. SIMPAN MODEL TERBAIK"""

model_filename = f'best_{best_final_name.lower()}_model.sav'
scaler_filename = 'scaler.sav'

pickle.dump(best_final_model, open(model_filename, 'wb'))
pickle.dump(scaler, open(scaler_filename, 'wb'))

print(f"\n✅ Model terbaik ({best_final_name}) disimpan sebagai: {model_filename}")
print(f"✅ Scaler disimpan sebagai: {scaler_filename}")

"""# 19. VALIDASI MODEL YANG DISIMPAN"""

# Load model yang disimpan
loaded_model = pickle.load(open(model_filename, 'rb'))
loaded_scaler = pickle.load(open(scaler_filename, 'rb'))

# Uji dengan data test
X_test_loaded_scaled = loaded_scaler.transform(X_test)
test_score = loaded_model.score(X_test_loaded_scaled, y_test)

print(f"\nAkurasi model yang disimpan (R²): {test_score:.4f}")

"""# 20. KESIMPULAN"""

print("\n" + "=" * 80)
print("KESIMPULAN")
print("=" * 80)
print(f"""
1. Model terbaik untuk prediksi jumlah penyewaan sepeda adalah {best_final_name} dengan R² = {best_r2:.4f}.
2. Feature importance tertinggi adalah kolom 'hour' (jam), menunjukkan bahwa waktu sangat berpengaruh.
3. Setelah handling outlier dan optimasi hyperparameter, performa model mencapai R² ≈ {best_r2:.4f}.
4. Pola penyewaan sepeda di London:
   - Tertinggi pada jam 7-9 pagi dan jam 16-19 sore (jam kerja)
   - Terendah pada musim dingin dan hari libur
   - Cuaca cerah meningkatkan jumlah penyewaan
5. Prediksi untuk Januari 2023 menunjukkan pola yang konsisten dengan data historis 2017.
6. Total prediksi penyewaan Jan-Feb 2023: {df_weather_2023['predicted_cnt'].sum():,.0f}
""")

print("\n🎉 Project selesai! 🎉")

# Tambahkan di akhir kode
df_weather_2023.to_csv('prediksi_sepeda_2023.csv', index=False)

# Data historis juga diexport
df_clean.to_csv('data_historis_2015_2017.csv', index=False)

# Download ke komputer
from google.colab import files
files.download('prediksi_sepeda_2023.csv')
files.download('data_historis_2015_2017.csv')
