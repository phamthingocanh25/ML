# (Lưu ý: Notebook của bạn dùng 1000 cho val+test, nhưng lại trừ 5 (horizon) 
# sau khi dropna (snippet 17, cell 48). 
# Để pipeline hoạt động đúng, ta phải trừ horizon ngay từ đầu)
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error # <-- Thêm import

# --- 1. Định nghĩa các tham số (dựa trên file notebook) ---
TARGET_COL = 'temp'
HORIZON = 5 

lag_cols = [
    'temp', 'humidity', 'dew', 'precip', 'precipprob', 'precipcover', 
    'solarradiation', 'sealevelpressure', 'windspeed', 'winddir', 
    'windgust', 'cloudcover', 'visibility'
]
lags = range(7, 91, 7) 

roll_cols = ['temp', 'humidity', 'windspeed', 'dew', 'cloudcover']
roll_windows = range(7, 91, 7) 

cols_to_drop_originals = [
    'name', 'datetime', 'sunrise', 'sunset', 
    'conditions', 'description', 'icon', 'stations', 'preciptype',
    'tempmax', 'tempmin'
]
cols_to_drop_features = list(set(lag_cols + roll_cols))
COLS_TO_DROP = sorted(list(set(cols_to_drop_originals + cols_to_drop_features)))


# --- 2. Định nghĩa các Transformer tùy chỉnh ---

class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    """
    Thực hiện 2 bước:
    1. Chuyển đổi các cột datetime (datetime, sunrise, sunset).
    2. Impute cột 'severerisk' (từ Cell 31, snippet 12).
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        
        for col in ['datetime', 'sunrise', 'sunset']:
            if col in X_.columns:
                X_[col] = pd.to_datetime(X_[col], errors='coerce')
        
        if 'severerisk' in X_.columns:
            X_['severerisk'] = X_['severerisk'].fillna(10) 
        
        return X_

class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    """
    (ĐÃ SỬA LỖI PERFORMANCE WARNING)
    Thực hiện tất cả các bước tạo đặc trưng mới (dựa trên Cell 38-42 và 47).
    """
    def __init__(self, lag_cols, lags, roll_cols, windows):
        self.lag_cols = lag_cols
        self.lags = lags
        self.roll_cols = roll_cols
        self.windows = windows
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X_ = X.copy()
        # Tạo một DataFrame tạm để chứa các feature mới (sửa lỗi PerformanceWarning)
        new_features_df = pd.DataFrame(index=X_.index)
            
        # 2.1 Cyclical (Cell 38 / 47)
        if 'datetime' in X_.columns:
            dt = X_['datetime'].dt
            new_features_df['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
            new_features_df['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
            new_features_df['day_of_year_sin'] = np.sin(2 * np.pi * dt.dayofyear / 365.25)
            new_features_df['day_of_year_cos'] = np.cos(2 * np.pi * dt.dayofyear / 365.25)
            new_features_df['day_of_week_sin'] = np.sin(2 * np.pi * dt.dayofweek / 7)
            new_features_df['day_of_week_cos'] = np.cos(2 * np.pi * dt.dayofweek / 7)
            new_features_df['quarter'] = dt.quarter
            new_features_df['year'] = dt.year

        # 2.2 Lags (Cell 39 / 47)
        for feature in self.lag_cols:
            if feature in X_.columns:
                for lag in self.lags:
                    new_features_df[f'{feature}_lag{lag}'] = X_[feature].shift(lag)

        # 2.3 Rolling (Cell 40 / 47)
        for feature in self.roll_cols:
             if feature in X_.columns:
                for w in self.windows:
                    # shift(1) để tránh data leakage
                    rolling_window = X_[feature].shift(1).rolling(window=w)
                    new_features_df[f'{feature}_roll{w}_mean'] = rolling_window.mean()
                    new_features_df[f'{feature}_roll{w}_std'] = rolling_window.std()
                    
        # 2.4 Derived (Cell 41, 42 / 47)
        if 'tempmax' in X_.columns and 'tempmin' in X_.columns:
            new_features_df['temp_range'] = X_['tempmax'] - X_['tempmin']
        if 'humidity' in X_.columns:
            new_features_df['humidity_change'] = X_['humidity'].diff()
        if 'sunrise' in X_.columns and 'sunset' in X_.columns:
            sr = X_['sunrise']
            ss = X_['sunset']
            valid_times = sr.notna() & ss.notna()
            # Tạo 1 Series tạm
            day_length_col = pd.Series(np.nan, index=X_.index, dtype='float64')
            if valid_times.any():
                day_length_col.loc[valid_times] = (ss[valid_times] - sr[valid_times]).dt.total_seconds() / 3600
            new_features_df['day_length_hour'] = day_length_col
        
        # Nối tất cả các feature mới vào X_ MỘT LẦN DUY NHẤT
        X_ = pd.concat([X_, new_features_df], axis=1)
        
        return X_

class ColumnCleanupTransformer(BaseEstimator, TransformerMixin):
    """
    Loại bỏ các cột gốc (đã được tạo feature) và các cột phi số.
    """
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop
        self.feature_names_ = [] 

    def fit(self, X, y=None):
        X_temp = X.drop(columns=self.cols_to_drop, errors='ignore')
        self.feature_names_ = X_temp.select_dtypes(include=np.number).columns.tolist()
        return self
        
    def transform(self, X, y=None):
        X_ = X.copy()
        X_ = X_.drop(columns=self.cols_to_drop, errors='ignore')
        
        # Đảm bảo chỉ các cột đã fit được giữ lại và đúng thứ tự
        X_ = X_[self.feature_names_] 
        return X_
        
    def get_feature_names_out(self, input_features=None):
         return np.array(self.feature_names_)

# --- 3. Tải dữ liệu và chuẩn bị X, y (Để fit pipeline) ---

print("Đang tải dữ liệu...")
url = 'https://raw.githubusercontent.com/DanhBitoo/HanoiDaily-temperature/refs/heads/main/Hanoi%20Daily.csv'
df_daily = pd.read_csv(url)

# Tạo các cột target (y)
df_y_full = pd.DataFrame(index=df_daily.index)
target_names = []
for i in range(1, HORIZON + 1):
    col_name = f'target_{TARGET_COL}_t+{i}'
    df_y_full[col_name] = df_daily[TARGET_COL].shift(-i)
    target_names.append(col_name)

df_X_full = df_daily

# --- 4. Tách Train/Val/Test (Dựa trên logic của Notebook) ---
n_val = 500
n_test = 500
n_val_split = n_val + HORIZON
n_test_split = n_test + HORIZON

df_train = df_X_full.iloc[:-n_val_split - n_test_split]
df_val = df_X_full.iloc[-n_val_split - n_test_split:-n_test_split]
df_test = df_X_full.iloc[-n_test_split:]

y_train = df_y_full.iloc[:-n_val_split - n_test_split]
y_val = df_y_full.iloc[-n_val_split - n_test_split:-n_test_split]
y_test = df_y_full.iloc[-n_test_split:]

# --- 5. Xử lý NaNs (Quan trọng) ---
y_train_nonan_idx = y_train.dropna().index
X_train_fit = df_train.loc[y_train_nonan_idx]
y_train_fit = y_train.loc[y_train_nonan_idx]

max_lookback = 90 
X_train_fit = X_train_fit.iloc[max_lookback:]
y_train_fit = y_train_fit.iloc[max_lookback:]

print(f"Dữ liệu X thô ban đầu (train): {df_train.shape}")
print(f"Dữ liệu X và y dùng để fit (đã bỏ NaNs đầu/cuối): {X_train_fit.shape}, {y_train_fit.shape}")


# --- 6. Định nghĩa Pipeline Lớn (ĐÃ SỬA LỖI ValueError) ---

# Đây là pipeline *bên trong* (inner) cho MỘT target
# Nó sẽ được MultiOutputRegressor gọi 5 lần.
inner_pipeline = Pipeline([
    # Bước 1: Chuyển dtype (datetime) và fill 'severerisk'
    ('preprocessing', PreprocessingTransformer()),
    
    # Bước 2: Tạo đặc trưng (cyclical, lag, rolling, derived)
    ('feature_engineering', FeatureEngineeringTransformer(
        lag_cols=lag_cols, 
        lags=lags, 
        roll_cols=roll_cols, 
        windows=roll_windows
    )),
    
    # Bước 3: Dọn dẹp cột (bỏ cột gốc, cột phi số)
    ('cleanup', ColumnCleanupTransformer(
        cols_to_drop=COLS_TO_DROP
    )),
    
    # Bước 4: Fill NaNs (tạo ra từ lag/roll ở đầu dữ liệu)
    ('imputer', SimpleImputer(strategy='median')), 
    
    # Bước 5: Scaling (từ Cell 51)
    ('scaler', StandardScaler()),
    
    # Bước 6: Feature Selection (từ Cell 51)
    # BƯỚC NÀY GIỜ SẼ NHẬN y 1 CHIỀU (vd: t+1) VÀ SẼ HOẠT ĐỘNG
    ('selector', SelectKBest(f_regression, k=50)),
    
    # Bước 7: Model (CHỈ DÙNG Ridge, KHÔNG PHẢI MultiOutputRegressor)
    ('model', Ridge(alpha=0.1))
])

# --- 7. Huấn luyện và Lưu Pipeline (ĐÃ SỬA LỖI ValueError) ---

# BỌC (WRAP) inner_pipeline BẰNG MultiOutputRegressor
# Model cuối cùng sẽ là 5 pipelines (đã fit) khác nhau, 
# mỗi pipeline cho một ngày dự đoán.
final_model_to_save = MultiOutputRegressor(inner_pipeline)

print("\nĐang huấn luyện 5 pipelines (bên trong MultiOutputRegressor)...")
# Huấn luyện trên dữ liệu đã được làm sạch index
final_model_to_save.fit(X_train_fit, y_train_fit)
print("Huấn luyện hoàn tất!")

# Lưu file
pipeline_filename = 'full_weather_pipeline.pkl'
# Lưu model bọc bên ngoài (MultiOutputRegressor)
joblib.dump(final_model_to_save, pipeline_filename) 

print(f"\nĐÃ LƯU PIPELINE HOÀN CHỈNH VÀO FILE: {pipeline_filename}")

# --- 8. (Tùy chọn) Kiểm tra pipeline ---
print("\nĐang kiểm tra dự đoán trên tập Val (dùng pipeline đã lưu)...")
# Tải lại model
loaded_model = joblib.load(pipeline_filename)

# Dự đoán trên df_val thô (chưa xử lý)
# Pipeline sẽ tự động xử lý NaNs (từ lag/roll/impute) bên trong nó
y_pred_val = loaded_model.predict(df_val)

# Chuyển y_pred (numpy array) thành DataFrame để dễ so sánh
y_pred_val_df = pd.DataFrame(y_pred_val, index=y_val.index, columns=target_names)

# Bỏ 90 hàng đầu của y_val và y_pred_val_df (do lag/roll)
# và 5 hàng cuối (do target shift) để so sánh MAE
y_val_clean = y_val.iloc[max_lookback:-HORIZON]
y_pred_val_clean = y_pred_val_df.iloc[max_lookback:-HORIZON]

if not y_val_clean.empty and not y_pred_val_clean.empty:
    mae_val = mean_absolute_error(y_val_clean, y_pred_val_clean)
    print(f"MAE trên tập Val (sử dụng pipeline .pkl): {mae_val:.4f}")
else:
    print(f"Không thể tính MAE, tập val/pred sau khi xử lý bị rỗng (y_val_clean: {y_val_clean.shape}, y_pred_val_clean: {y_pred_val_clean.shape}).")