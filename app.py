import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
# THÊM IMPORT NÀY: Cần thiết cho các lớp tùy chỉnh
from sklearn.base import BaseEstimator, TransformerMixin 

# --- SAO CHÉP CÁC ĐỊNH NGHĨA LỚP TỪ FILE HUẤN LUYỆN VÀO ĐÂY ---

class PreprocessingTransformer(BaseEstimator, TransformerMixin):
    """
    Thực hiện 2 bước:
    1. Chuyển đổi các cột datetime (datetime, sunrise, sunset).
    2. Impute cột 'severerisk'.
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
    Thực hiện tất cả các bước tạo đặc trưng mới.
    (Lưu ý: các tham số lag_cols, lags... sẽ được nạp từ file pkl)
    """
    def __init__(self, lag_cols=None, lags=None, roll_cols=None, windows=None):
        self.lag_cols = lag_cols
        self.lags = lags
        self.roll_cols = roll_cols
        self.windows = windows
    
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        X_ = X.copy()
        new_features_df = pd.DataFrame(index=X_.index)
            
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

        if self.lag_cols and self.lags:
            for feature in self.lag_cols:
                if feature in X_.columns:
                    for lag in self.lags:
                        new_features_df[f'{feature}_lag{lag}'] = X_[feature].shift(lag)

        if self.roll_cols and self.windows:
            for feature in self.roll_cols:
                 if feature in X_.columns:
                    for w in self.windows:
                        rolling_window = X_[feature].shift(1).rolling(window=w)
                        new_features_df[f'{feature}_roll{w}_mean'] = rolling_window.mean()
                        new_features_df[f'{feature}_roll{w}_std'] = rolling_window.std()
                    
        if 'tempmax' in X_.columns and 'tempmin' in X_.columns:
            new_features_df['temp_range'] = X_['tempmax'] - X_['tempmin']
        if 'humidity' in X_.columns:
            new_features_df['humidity_change'] = X_['humidity'].diff()
        if 'sunrise' in X_.columns and 'sunset' in X_.columns:
            sr = X_['sunrise']
            ss = X_['sunset']
            valid_times = sr.notna() & ss.notna()
            day_length_col = pd.Series(np.nan, index=X_.index, dtype='float64')
            if valid_times.any():
                day_length_col.loc[valid_times] = (ss[valid_times] - sr[valid_times]).dt.total_seconds() / 3600
            new_features_df['day_length_hour'] = day_length_col
        
        X_ = pd.concat([X_, new_features_df], axis=1)
        return X_

class ColumnCleanupTransformer(BaseEstimator, TransformerMixin):
    """
    Loại bỏ các cột gốc (đã được tạo feature) và các cột phi số.
    """
    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop
        self.feature_names_ = [] 

    def fit(self, X, y=None):
        # Đảm bảo cols_to_drop là một danh sách
        cols_to_drop_safe = self.cols_to_drop if self.cols_to_drop is not None else []
        X_temp = X.drop(columns=cols_to_drop_safe, errors='ignore')
        self.feature_names_ = X_temp.select_dtypes(include=np.number).columns.tolist()
        return self
        
    def transform(self, X, y=None):
        X_ = X.copy()
        # Đảm bảo cols_to_drop là một danh sách
        cols_to_drop_safe = self.cols_to_drop if self.cols_to_drop is not None else []
        X_ = X_.drop(columns=cols_to_drop_safe, errors='ignore')
        
        # Xử lý trường hợp các cột bị thiếu (ví dụ: khi dự đoán trên dữ liệu mới)
        # Chỉ giữ lại các cột đã thấy lúc fit, và thêm các cột bị thiếu (nếu có) với giá trị np.nan
        X_out = pd.DataFrame(index=X_.index)
        for col in self.feature_names_:
            if col in X_.columns:
                X_out[col] = X_[col]
            else:
                # Cột này có lúc fit nhưng không có lúc transform
                X_out[col] = np.nan 
                
        return X_out[self.feature_names_] # Đảm bảo đúng thứ tự cột
        
    def get_feature_names_out(self, input_features=None):
         return np.array(self.feature_names_)

# --- KẾT THÚC PHẦN SAO CHÉP LỚP ---


# --- 1. Cấu hình trang (Giữ nguyên) ---
st.set_page_config(
    page_title="Dự đoán thời tiết Hà Nội",
    page_icon="🌤️",
    layout="wide"
)

# --- 2. Hằng số (Giữ nguyên) ---
MODEL_PATH = 'full_weather_pipeline.pkl'
DATA_URL = 'https://raw.githubusercontent.com/DanhBitoo/HanoiDaily-temperature/refs/heads/main/Hanoi%20Daily.csv'
HORIZON = 5 

# --- 3. Hàm tải mô hình (Giữ nguyên) ---
@st.cache_resource
def load_model(path):
    """
    Tải mô hình pipeline từ file .pkl.
    Sử dụng cache_resource để chỉ tải mô hình một lần.
    """
    try:
        model = joblib.load(path)
        return model
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file mô hình tại '{path}'.")
        st.error("Vui lòng đảm bảo file `full_weather_pipeline.pkl` nằm cùng thư mục với `app.py`.")
        return None
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.exception(e) # In ra traceback đầy đủ
        return None

# --- 4. Hàm tải dữ liệu (Giữ nguyên) ---
@st.cache_data
def load_data(url):
    """
    Tải dữ liệu lịch sử từ URL.
    Sử dụng cache_data để chỉ tải dữ liệu một lần.
    """
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu từ URL: {e}")
        return None

# --- 5. Giao diện chính của ứng dụng (Giữ nguyên) ---
def main():
    st.title("🌤️ Ứng dụng dự đoán nhiệt độ 5 ngày tới tại Hà Nội")
    st.write("Ứng dụng này sử dụng mô hình Ridge Regression đã huấn luyện (bao gồm toàn bộ pipeline xử lý) để dự đoán nhiệt độ.")

    # Tải mô hình
    model = load_model(MODEL_PATH)
    if model is None:
        st.stop() 

    # Tải dữ liệu
    df_history = load_data(DATA_URL)
    if df_history is None:
        st.stop() 

    # Hiển thị thông tin dữ liệu
    st.subheader("Dữ liệu lịch sử (mới nhất)")
    st.write(f"Đã tải {len(df_history)} ngày dữ liệu lịch sử từ GitHub.")
    
    try:
        last_date_str = df_history['datetime'].iloc[-1]
        last_date = pd.to_datetime(last_date_str)
        st.info(f"Dữ liệu lịch sử mới nhất là của ngày: **{last_date.strftime('%Y-%m-%d')}**")
    except Exception as e:
        st.error(f"Lỗi khi xử lý cột 'datetime': {e}")
        st.dataframe(df_history.tail())
        st.stop()
        
    st.dataframe(df_history.tail())

    # Nút dự đoán
    st.subheader("Bắt đầu dự đoán")
    if st.button(f"Dự đoán {HORIZON} ngày tiếp theo", type="primary"):
        
        with st.spinner("Đang chạy pipeline... (tính toán cyclical, lags, rolling, scaling... và dự đoán)"):
            try:
                # Đưa TOÀN BỘ dữ liệu lịch sử thô vào hàm predict.
                all_predictions = model.predict(df_history)
                
                # Dự đoán chúng ta cần nằm ở HÀNG CUỐI CÙNG
                future_predictions = all_predictions[-1]

                st.success("Dự đoán hoàn tất!")

                # Tạo các ngày trong tương lai
                future_dates = [last_date + timedelta(days=i) for i in range(1, HORIZON + 1)]
                
                # Tạo DataFrame kết quả
                df_results = pd.DataFrame({
                    'Ngày dự đoán': future_dates,
                    f'Nhiệt độ dự đoán (°C)': future_predictions
                })
                
                # Hiển thị kết quả
                st.subheader("Kết quả dự đoán")
                
                df_results_display = df_results.copy()
                df_results_display['Ngày dự đoán'] = df_results_display['Ngày dự đoán'].dt.strftime('%Y-%m-%d')
                
                st.dataframe(df_results_display, use_container_width=True)

                # Hiển thị biểu đồ
                st.subheader("Biểu đồ dự đoán")
                
                chart_data = df_results.set_index('Ngày dự đoán')
                st.line_chart(chart_data)

            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
                st.exception(e) 

# Chạy ứng dụng
if __name__ == "__main__":
    main()

