import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Cấu hình trang ---
st.set_page_config(
    page_title="🌤️ Dự đoán thời tiết Hà Nội",
    page_icon="🌦️",
    layout="wide"
)

# --- Đường dẫn ---
MODEL_PATH = "multi_rf_final_using_X_final.pkl"
SELECTOR_PATH = r"C:\Users\Admin\Downloads\ML của vk\multi_selector.pkl"
DATA_PATH = r"C:\Users\Admin\Downloads\ML của vk\X_train_final.csv"
HORIZON = 5

# --- Hàm tải mô hình ---
@st.cache_resource
def load_models():
    try:
        model = joblib.load(MODEL_PATH)
        st.success("✅ Đã tải multi_rf.pkl thành công.")
    except Exception as e:
        st.error(f"Lỗi khi tải multi_rf.pkl: {e}")
        return None, None

    try:
        selector = joblib.load(SELECTOR_PATH)
        st.success("✅ Đã tải multi_selector.pkl thành công.")
    except Exception as e:
        st.warning("⚠️ Không tìm thấy multi_selector.pkl — sẽ bỏ qua bước chọn đặc trưng.")
        selector = None

    return model, selector

# --- Hàm tải dữ liệu ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Lỗi khi đọc file CSV: {e}")
        return None

# --- Ứng dụng chính ---
def main():
    st.title("🌤️ Ứng dụng dự đoán nhiệt độ 5 ngày tới tại Hà Nội")
    st.write("Ứng dụng sử dụng mô hình **Random Forest đa đầu ra (MultiOutputRegressor)** để dự đoán nhiều giá trị thời tiết.")

    model, selector = load_models()
    if model is None:
        st.stop()

    df = load_data()
    if df is None:
        st.stop()

    # --- Hiển thị thông tin dữ liệu ---
    st.subheader("📘 Dữ liệu lịch sử")
    st.dataframe(df.tail(), use_container_width=True)

    # Xác định cột thời gian
    date_col = None
    for c in ["datetime", "date", "day"]:
        if c in df.columns:
            date_col = c
            break

    if date_col:
        try:
            last_date = pd.to_datetime(df[date_col].iloc[-1])
            st.info(f"Dữ liệu mới nhất: **{last_date.strftime('%Y-%m-%d')}**")
        except Exception:
            last_date = pd.Timestamp.today()
    else:
        last_date = pd.Timestamp.today()

    # --- Chọn cột đầu vào ---
    X = df.select_dtypes(include=[np.number]).copy()
    st.write(f"🧮 Tổng số đặc trưng: {X.shape[1]}")

    # --- Nút dự đoán ---
    st.subheader("🔮 Dự đoán")
    if st.button(f"Dự đoán {HORIZON} ngày tiếp theo", type="primary"):
        with st.spinner("Đang dự đoán..."):
            try:
                # Nếu có selector thì apply trước
                if selector is not None:
                    X_sel = selector.transform(X)
                else:
                    X_sel = X

                preds = model.predict(X_sel)
                if isinstance(preds, np.ndarray) and preds.ndim > 1:
                    future_preds = preds[-1]  # lấy hàng cuối cùng
                else:
                    future_preds = [preds[-1]]

                future_dates = [last_date + timedelta(days=i) for i in range(1, HORIZON + 1)]

                # Hiển thị kết quả
                df_result = pd.DataFrame({
                    "Ngày": future_dates,
                    "Nhiệt độ dự đoán (°C)": future_preds[:HORIZON]
                })

                st.success("✅ Dự đoán hoàn tất!")
                st.dataframe(df_result, use_container_width=True)

                # --- Vẽ biểu đồ bằng Matplotlib ---
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_result['Ngày'], df_result['Nhiệt độ dự đoán (°C)'], marker='o', linestyle='-')

                # Giới hạn trục Y theo min–max dự đoán ±0.5°C
                y_min = df_result['Nhiệt độ dự đoán (°C)'].min() - 0.5
                y_max = df_result['Nhiệt độ dự đoán (°C)'].max() + 0.5
                ax.set_ylim(y_min, y_max)

                # Trang trí biểu đồ
                ax.set_xlabel('Ngày')
                ax.set_ylabel('Nhiệt độ (°C)')
                ax.set_title('Dự đoán nhiệt độ 5 ngày tới')
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                fig.autofmt_xdate()

                st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Lỗi khi dự đoán: {e}")
                st.exception(e)

if __name__ == "__main__":
    main()
