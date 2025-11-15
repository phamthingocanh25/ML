import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc 
import pandas as pd
import base64
import os
import io
import json # Import thêm json

# ===== Tên file data dự đoán =====
DATA_FILE = "forecast_results.json"

# ===== HÀM ĐỌC ẢNH (Gần giống code cũ) =====
def get_base64_icon(icon_name):
    path = os.path.join('assets', icon_name)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode()

# ===== HÀM ĐỌC DỮ LIỆU TỪ FILE JSON (Gần giống code cũ) =====
def load_forecast_data(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Không tìm thấy file {filepath}")
        return []
    try:
        df = pd.read_json(filepath)
        df['date'] = pd.to_datetime(df['date'])
        data = []
        for _, row in df.iterrows():
            data.append({
                "day": row['date'].strftime('%a'),
                "date": row['date'].strftime('%d %b'),
                "temp": f"{row['forecast_temp_rf']:.2f}°C",
                "icon": "☀️"
            })
        return data
    except Exception as e:
        print(f"Lỗi khi đọc {filepath}: {e}")
        return []

# ===== KHỞI TẠO ỨNG DỤNG DASH =====
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ===== HÀM TẠO CARD HIỂN THỊ CHÍNH (Ô LỚN) =====
def create_main_display(day, date, temp, icon):
    return html.Div(className="summary-card", children=[
        html.Div(f"{icon} Clear", style={"fontSize": "22px", "opacity": "0.8"}),
        html.Div(f"{day}, {date}", className="hanoi-font", style={"fontSize": "26px", "fontWeight": "700"}),
        html.Div(temp, style={"fontSize": "80px", "fontWeight": "700", "marginTop": "-5px"}),
        html.Div("Hanoi, Vietnam", style={"opacity": "0.8", "marginTop": "5px"})
    ])

# ===== LAYOUT CỦA ỨNG DỤNG (HTML) =====
app.layout = html.Div(id="main-container", children=[
    
    dcc.Store(id='forecast-data-store'),
    html.Div(className="bg-slide bg1"),
    html.Div(className="bg-slide bg2"),
    html.Div(className="bg-slide bg3"),
    html.Div(className="bg-slide bg4"),
    html.Div(className="bg-slide bg5"),
    
    html.Div(
        id="main-display-panel",
        className="left-panel",
        children=[], 
        style={'display': 'none'} 
    ),
    
    html.Div(
        id="display-button-container",
        children=[
            html.Button(
                
                children=[
                    html.Div("Weather Forcast", className="button-title-main"),
                    html.Div("Hanoi,Vietnam", className="button-title-sub")
                ],
                id="display-forecast-button",
                n_clicks=0
            )
        ]
    ),
    
    html.Div(
        id="forecast-container",
        style={'display': 'none'} 
    )
])

# ===== CALLBACKS =====
@app.callback(
    Output('forecast-data-store', 'data'),
    Output('forecast-container', 'children'),
    Output('main-display-panel', 'children', allow_duplicate=True),
    Output('main-display-panel', 'style'), 
    Output('forecast-container', 'style'), 
    Output('display-button-container', 'style'),
    Input('display-forecast-button', 'n_clicks'),
    prevent_initial_call=True
)
def show_forecast_cards(n_clicks):
    # 1. Tải dữ liệu
    data = load_forecast_data(DATA_FILE)
    
    if not data:
        # Nếu không có data, không làm gì cả
        return (None, "Không tìm thấy data", dash.no_update, dash.no_update, dash.no_update, dash.no_update)

    # 2. Tạo 5 ô nhỏ
    small_cards = []
    for i, day_data in enumerate(data):
        small_cards.append(
            html.Button(
                className="forecast-card-button",
                children=[
                    html.Div(day_data['day'], className="card-day"),
                    html.Div(day_data['date'], className="card-date"),
                    html.Div(day_data['temp'], className="card-temp"),
                    html.Div(day_data['icon'], className="card-icon")
                ],
                id={
                    'type': 'day-forecast-button',
                    'index': i
                },
                n_clicks=0
            )
        )
    
    first_day_data = data[0]
    main_display = create_main_display(
        first_day_data['day'],
        first_day_data['date'],
        first_day_data['temp'],
        first_day_data['icon']
    )

    # 4. Trả về kết quả
    panel_style_show = {'display': 'block'} # SỬA: Style để hiện ô lớn
    forecast_style_show = {'display': 'flex'} # SỬA: Style để hiện 5 ô nhỏ (dùng flex)
    style_hide = {'display': 'none'}  # Style để ẩn nút
    
    # Trả về data, 5 ô nhỏ, ô lớn, style (hiện), style (hiện), style (ẩn)
    # THAY ĐỔI THỨ TỰ TRẢ VỀ STYLE CHO ĐÚNG:
    return (data, small_cards, main_display, panel_style_show, forecast_style_show, style_hide)
    

# --- Callback 2: Khi nhấn 1 trong 5 ô nhỏ ---
@app.callback(
    Output('main-display-panel', 'children', allow_duplicate=True), 
    Input({'type': 'day-forecast-button', 'index': ALL}, 'n_clicks'),
    State('forecast-data-store', 'data'),
    prevent_initial_call=True # Thêm dòng này để callback không chạy lúc mở trang
)
def update_main_display(n_clicks_list, data):
    
    ctx = dash.callback_context
    if not ctx.triggered or not data:
        return dash.no_update

    # Lấy thông tin của nút vừa được nhấn
    button_id_str = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Check nếu button_id_str rỗng (đôi khi xảy ra)
    if not button_id_str:
        return dash.no_update
        
    try:
        button_id_dict = json.loads(button_id_str)
        clicked_index = button_id_dict['index']
    except Exception as e:
        print(f"Lỗi khi parse ID: {e}")
        return dash.no_update

    # Lấy data của ngày tương ứng
    selected_day_data = data[clicked_index]
    
    # Tạo ô hiển thị lớn mới và trả về
    main_display = create_main_display(
        selected_day_data['day'],
        selected_day_data['date'],
        selected_day_data['temp'],
        selected_day_data['icon']
    )
    
    return main_display

# ===== CHẠY SERVER =====
if __name__ == '__main__':
    # Dùng app.run thay vì app.run_server
    app.run(debug=True)