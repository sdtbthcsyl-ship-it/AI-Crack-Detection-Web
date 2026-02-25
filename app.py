import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
import pandas as st_pandas
import pandas as pd
import io

# ==========================================
# 1. CẤU HÌNH TRANG & GIAO DIỆN (CSS TÙY CHỈNH)
# ==========================================
st.set_page_config(
    page_title="Hệ thống AI Kiểm định Công trình",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Nhúng CSS để giao diện mượt mà và sinh động hơn
st.markdown("""
    <style>
    /* Chỉnh màu nền và font chữ */
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #1e3a8a; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    
    /* Làm đẹp các thẻ Metrics (Số liệu) */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e2e8f0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Nút bấm nổi bật */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. KHỞI TẠO MÔ HÌNH (CÓ BỘ NHỚ ĐỆM)
# ==========================================
@st.cache_resource
def load_yolo_model():
    model_path = 'best.pt'
    if os.path.exists(model_path):
        return YOLO(model_path)
    else:
        return None

# ==========================================
# 3. THANH ĐIỀU KHIỂN BÊN TRÁI (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103468.png", width=80)
    st.title("⚙️ Bảng Điều Khiển")
    st.markdown("---")
    
    # Lựa chọn nguồn ảnh (Upload hoặc Webcam mô phỏng UAV)
    input_source = st.radio("Lựa chọn nguồn dữ liệu:", ("Tải ảnh lên (Local)", "Sử dụng Camera (UAV Demo)"))
    
    st.markdown("---")
    st.subheader("🎛️ Tinh chỉnh AI")
    # Thanh trượt độ nhạy: Rất quan trọng khi chạy thực tế
    conf_thresh = st.slider("Ngưỡng tin cậy (Confidence)", min_value=0.05, max_value=1.00, value=0.20, step=0.05, 
                            help="Giảm số này nếu AI bỏ sót vết nứt mờ. Tăng lên nếu AI nhận diện nhầm.")
    
    st.markdown("---")

# ==========================================
# 4. GIAO DIỆN CHÍNH (HEADER & UPLOAD)
# ==========================================
st.title("🏗️ Nền tảng AI Phát hiện Vết nứt Bề mặt Công trình")
st.markdown("Hệ thống phân tích hình ảnh ứng dụng **Deep Learning (YOLOv8)** để tự động đánh giá an toàn kết cấu.")

model = load_yolo_model()

if model is None:
    st.error("❌ Lỗi nghiêm trọng: Không tìm thấy file `best.pt`. Vui lòng copy file từ Google Colab vào thư mục dự án.")
    st.stop() # Dừng chạy app nếu không có não bộ

image_to_process = None

# Xử lý nguồn dữ liệu đầu vào
if input_source == "Tải ảnh lên (Local)":
    uploaded_file = st.file_uploader("Kéo thả hoặc chọn ảnh bề mặt bê tông/tường tại đây...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image_to_process = Image.open(uploaded_file)
else:
    camera_file = st.camera_input("Chụp ảnh bề mặt cần kiểm tra")
    if camera_file:
        image_to_process = Image.open(camera_file)

# ==========================================
# 5. XỬ LÝ LÕI VÀ HIỂN THỊ KẾT QUẢ KHOA HỌC
# ==========================================
if image_to_process:
    st.markdown("---")
    
    # Nút bấm trung tâm
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        analyze_clicked = st.button("🚀 BẮT ĐẦU QUÉT AI & PHÂN TÍCH TIẾT DIỆN", use_container_width=True)

    if analyze_clicked:
        with st.spinner("🧠 Khởi động mạng Neural... Đang quét ma trận điểm ảnh..."):
            
            # Chuyển ảnh cho AI đọc
            img_array = np.array(image_to_process)
            
            # Chạy Inference (Dự đoán)
            results = model.predict(source=img_array, conf=conf_thresh, save=False)
            boxes = results[0].boxes
            num_cracks = len(boxes)
            
            # Vẽ hình
            res_img = results[0].plot(line_width=2) 
            
            # ----------------------------------------
            # HIỂN THỊ TỔNG QUAN (METRICS)
            # ----------------------------------------
            st.subheader("📊 Báo cáo Tổng quan")
            m1, m2, m3 = st.columns(3)
            m1.metric(label="Tổng số vết nứt phát hiện", value=f"{num_cracks} vị trí", delta="Nguy cơ" if num_cracks > 0 else "An toàn", delta_color="inverse")
            
            if num_cracks > 0:
                max_conf = float(max(boxes.conf)) * 100
                m2.metric(label="Độ tin cậy cao nhất", value=f"{max_conf:.1f}%")
                m3.metric(label="Trạng thái", value="Cần kiểm tra lại", delta="⚠️ Cảnh báo", delta_color="inverse")
            else:
                m2.metric(label="Độ tin cậy", value="N/A")
                m3.metric(label="Trạng thái", value="Đạt chuẩn an toàn", delta="✅ Hoàn hảo")

            # ----------------------------------------
            # CHIA TABS ĐỂ TRÌNH BÀY SINH ĐỘNG
            # ----------------------------------------
            tab1, tab2, tab3 = st.tabs(["👁️ Trực quan hóa Hình ảnh", "📋 Bảng Dữ liệu Tọa độ", "⚙️ Thông số Kỹ thuật"])
            
            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**📸 Ảnh gốc ban đầu**")
                    st.image(image_to_process, use_container_width=True)
                with c2:
                    st.markdown("**🤖 Kết quả khoanh vùng bởi AI**")
                    st.image(res_img, use_container_width=True)

            with tab2:
                if num_cracks > 0:
                    st.markdown("Bảng chi tiết các vị trí bị tổn thương trên bề mặt, phục vụ cho việc lập hồ sơ bảo trì:")
                    
                    # Trích xuất dữ liệu khoa học
                    data = []
                    for i, box in enumerate(boxes):
                        conf = float(box.conf[0]) * 100
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height # Diện tích pixel (phục vụ đánh giá mức độ nứt)
                        
                        data.append({
                            "ID": f"Nứt-{i+1:02d}",
                            "Độ tin cậy (%)": round(conf, 2),
                            "Tọa độ X (pixel)": int(x1),
                            "Tọa độ Y (pixel)": int(y1),
                            "Chiều rộng (w)": int(width),
                            "Chiều cao (h)": int(height),
                            "Diện tích vùng tổn thương": int(area)
                        })
                    
                    # Tạo Pandas DataFrame
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    # Tính năng xuất file CSV cho báo cáo NCKH
                    csv = df.to_csv(index=False).encode('utf-8-sig')
                    st.download_button(
                        label="📥 Tải Bảng Dữ Liệu Báo Cáo (.CSV)",
                        data=csv,
                        file_name='baocao_kiemdinh_vetnut.csv',
                        mime='text/csv',
                    )
                else:
                    st.success("Bề mặt đồng nhất, không trích xuất dữ liệu tổn thương.")

            with tab3:
                st.markdown("""
                ### Thông tin Hệ thống Phân tích
                * **Thuật toán cốt lõi:** YOLOv8 (You Only Look Once) Mạng nơ-ron tích chập (CNN).
                * **Đầu vào xử lý (Input Shape):** Resize nội suy về 640x640 tensor.
                * **Phân lớp (Classes):** `[0] Crack` (Vết nứt bề mặt).
                * **Ngưỡng sàng lọc (NMS/Conf):** Tùy chỉnh trực tiếp qua thanh điều khiển Sidebar.
                * **Ứng dụng:** Tích hợp trên Payload của phương tiện bay không người lái (UAV) để truyền ảnh và phân tích Real-time.
                """)