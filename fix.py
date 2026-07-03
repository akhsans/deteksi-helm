import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import sqlite3
import bcrypt

st.set_page_config(page_title="HelmDetect", page_icon="🪖", layout="wide")

# ── Database ──────────────────────────────────────────────────────────────────
conn = sqlite3.connect("helmet.db", check_same_thread=False)
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT
    )
""")
conn.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id       INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT,
        time     TEXT,
        result   TEXT,
        image    BLOB
    )
""")
conn.commit()

# ── Migrasi: tambah kolom username jika belum ada (kompatibel dengan DB lama) ─
existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(history)").fetchall()]
if "username" not in existing_cols:
    conn.execute("ALTER TABLE history ADD COLUMN username TEXT")
    conn.commit()

# ── Auth helpers ──────────────────────────────────────────────────────────────
def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def register_user(username, password):
    try:
        conn.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hash_password(password))
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    row = conn.execute(
        "SELECT password FROM users WHERE username=?", (username,)
    ).fetchone()
    return row and check_password(password, row[0])

# ── Session state ─────────────────────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None

# ═════════════════════════════════════ HALAMAN LOGIN ═════════════════════════════════════════
if st.session_state.user is None:
    st.title("🔐 Login / Register — HelmDetect")

    tab_login, tab_reg = st.tabs(["Login", "Register"])

    with tab_login:
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            if login_user(username, password):
                st.session_state.user = username
                st.success("Login berhasil!")
                st.rerun()
            else:
                st.error("Username atau password salah")

    with tab_reg:
        new_user = st.text_input("Buat Username", key="reg_user")
        new_pass = st.text_input("Buat Password", type="password", key="reg_pass")
        if st.button("Register"):
            if register_user(new_user, new_pass):
                st.success("Berhasil register, silakan login")
            else:
                st.error("Username sudah digunakan")

    st.stop()

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    import torch
    # Izinkan objek internal YOLO agar tidak diblokir oleh sistem keamanan PyTorch
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except Exception:
        pass
        
    return YOLO("best.pt")

model = load_model()

# ── Helper: detect with custom box colors ─────────────────────────────────────
# Biru (BGR) untuk helm, Merah (BGR) untuk tanpa helm
COLOR_HELMET    = (255, 80,  80)   # biru
COLOR_NO_HELMET = (80,  80, 255)   # merah

def detect_image(image_rgb, conf, imgsz):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    results   = model.predict(image_bgr, conf=conf, imgsz=imgsz, verbose=False)

    annotated = image_bgr.copy()
    labels    = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id]
        labels.append(label)

        is_no_helmet = "no" in label.lower() or "without" in label.lower()
        color = COLOR_NO_HELMET if is_no_helmet else COLOR_HELMET

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf_val = float(box.conf[0])

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf_val:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    result_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return result_img, labels

def label_summary(labels):
    if not labels:
        return "Tidak terdeteksi"
    helm    = sum(1 for l in labels if "no" not in l.lower() and "without" not in l.lower())
    no_helm = sum(1 for l in labels if "no" in l.lower() or "without" in l.lower())
    parts   = []
    if helm:    parts.append(f"{helm} Pakai Helm")
    if no_helm: parts.append(f"{no_helm} Tanpa Helm")
    return ", ".join(parts)

def save_history(result_text, img_rgb):
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    conn.execute(
        "INSERT INTO history (username, time, result, image) VALUES (?, ?, ?, ?)",
        (st.session_state.user, datetime.now().strftime("%H:%M:%S"),
         result_text, buf.tobytes())
    )
    conn.commit()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📋 Menu")
menu = st.sidebar.radio("Pilih Halaman", ["🔍 Deteksi", "📜 Riwayat", "🪖 Deskripsi"])
st.sidebar.divider()

if st.sidebar.button("🚪 Logout"):
    st.session_state.user = None
    st.rerun()
# ═══════════════════════════════════ HALAMAN DETEKSI ══════════════════════════════
if menu == "🔍 Deteksi":
    st.title("🪖 HelmDetect")
    st.caption(f"Deteksi Penggunaan Helm · Login sebagai **{st.session_state.user}**")
    st.divider()

    tab_upload, tab_video, tab_realtime = st.tabs(["📁 Upload Gambar", "🎬 Video", "⚡ Realtime"])

    # ── Tab Upload Gambar ─────────────────────────────────────────────────────
    with tab_upload:
        col_set1, col_set2 = st.columns(2)
        with col_set1:
            conf_img = st.slider("Confidence Threshold", 0.1, 0.95, 0.50, 0.05, key="conf_img")
        with col_set2:
            imgsz_img = st.select_slider(
                "Image Size",
                options=[320, 416, 512, 608, 640, 736, 800],
                value=640,
                key="imgsz_img"
            )

        uploaded = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.cvtColor(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            st.image(img, caption="Gambar Input", use_container_width=True)

            if st.button("🔍 Deteksi Gambar"):
                with st.spinner("Mendeteksi..."):
                    result_img, labels = detect_image(img, conf_img, imgsz_img)
                    summary = label_summary(labels)

                st.image(result_img, caption="Hasil", use_container_width=True)

                if labels:
                    no_helm_ada = any("no" in l.lower() or "without" in l.lower() for l in labels)
                    if no_helm_ada:
                        st.error(f"⚠️ {summary}")
                    else:
                        st.success(f"✅ {summary}")
                else:
                    st.warning("Tidak ada objek terdeteksi")

                save_history(summary, result_img)

    # ── Tab Video ─────────────────────────────────────────────────────────────
    with tab_video:
        col_set3, col_set4 = st.columns(2)
        with col_set3:
            conf_vid = st.slider("Confidence Threshold", 0.1, 0.95, 0.30, 0.05, key="conf_vid")
        with col_set4:
            imgsz_vid = st.select_slider(
                "Image Size",
                options=[320, 416, 512, 608, 640, 736, 800],
                value=640,
                key="imgsz_vid"
            )

        uploaded_vid = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

        if uploaded_vid:
            import tempfile, os, glob

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_vid.read())
            tfile.flush()
            tfile.close()  # ← tutup handle dulu sebelum diproses

            st.video(tfile.name)

            if st.button("🔍 Deteksi Video"):
                try:
                    with st.spinner("⏳ Memproses video... Harap tunggu."):
                        out_dir = tempfile.mkdtemp()
                        model.predict(
                            source=tfile.name,
                            conf=conf_vid,
                            save=True,
                            project=out_dir,
                            name="hasil",
                            verbose=False
                        )

                    hasil_files = glob.glob(os.path.join(out_dir, "hasil", "*.mp4")) + \
                                  glob.glob(os.path.join(out_dir, "hasil", "*.avi"))

                    if hasil_files:
                        hasil_path = hasil_files[0]
                        st.success("✅ Deteksi selesai!")
                        st.video(hasil_path)

                        with open(hasil_path, "rb") as f:
                            st.download_button(
                                "⬇️ Download Video Hasil",
                                data=f.read(),
                                file_name="hasil_deteksi.mp4",
                                mime="video/mp4"
                            )

                        cap   = cv2.VideoCapture(hasil_path)
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
                        ret, last_frame = cap.read()
                        cap.release()
                        if ret:
                            last_rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                            save_history(f"Deteksi video: {uploaded_vid.name}", last_rgb)
                    else:
                        st.error("❌ Output video tidak ditemukan. Coba lagi.")

                finally:
                    try:
                        os.unlink(tfile.name)
                    except (PermissionError, FileNotFoundError):
                        pass  # Windows: file masih terkunci, skip

    # ── Tab Realtime ──────────────────────────────────────────────────────────
    with tab_realtime:
        col_set5, col_set6 = st.columns(2)
        with col_set5:
            conf_rt = st.slider("Confidence Threshold", 0.1, 0.95, 0.50, 0.05, key="conf_rt")
        with col_set6:
            imgsz_rt = st.select_slider(
                "Image Size",
                options=[320, 416, 512, 608, 640, 736, 800],
                value=640,
                key="imgsz_rt"
            )

        st.markdown("**Deteksi Realtime via Webcam**")
        st.caption("Streaming langsung — tiap frame dideteksi otomatis tanpa klik")

        # CSS video memenuhi lebar penuh container
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlock"] video {
            width: 100% !important;
            max-width: 100% !important;
            height: auto !important;
            border-radius: 8px;
        }
        iframe[title="streamlit_webrtc.frontend"] {
            width: 100% !important;
            min-height: 500px !important;
            border-radius: 8px;
        }
        </style>
        """, unsafe_allow_html=True)

        try:
            from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
            import av

            RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

            class HelmProcessor(VideoProcessorBase):
                def __init__(self):
                    self.conf  = conf_rt
                    self.imgsz = imgsz_rt

                def recv(self, frame):
                    img_bgr  = frame.to_ndarray(format="bgr24")
                    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    result_rgb, _ = detect_image(img_rgb, self.conf, self.imgsz)
                    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
                    return av.VideoFrame.from_ndarray(result_bgr, format="bgr24")

            webrtc_streamer(
                key="helmet-realtime",
                video_processor_factory=HelmProcessor,
                rtc_configuration=RTC_CONFIG,
                media_stream_constraints={
                    "video": {
                        "width":  {"ideal": 1280},
                        "height": {"ideal": 720},
                    },
                    "audio": False,
                },
            )

            st.info("💡 Klik **START** di atas untuk memulai. Bounding box langsung muncul di video.")

        except ImportError:
            st.error("⚠️ Library `streamlit-webrtc` belum terinstall.")
            st.code("pip install streamlit-webrtc av", language="bash")

# ═════════════════════════════════════════ HALAMAN RIWAYAT ═════════════════════════════════════════
elif menu == "📜 Riwayat":
    st.title("📜 Riwayat Deteksi")
    st.caption(f"Login sebagai **{st.session_state.user}**")

    data = conn.execute(
        "SELECT time, result, image FROM history WHERE username=? ORDER BY id DESC",
        (st.session_state.user,)
    ).fetchall()

    if not data:
        st.info("Belum ada riwayat deteksi")
    else:
        cols = st.columns(3)
        for i, item in enumerate(data):
            image = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(item[2], np.uint8), 1),
                cv2.COLOR_BGR2RGB
            )
            with cols[i % 3]:
                st.image(image, use_container_width=True)
                st.caption(f"🕒 {item[0]}")
                st.write(item[1])

        st.divider()
        if st.button("🗑️ Hapus Semua Riwayat"):
            conn.execute("DELETE FROM history WHERE username=?", (st.session_state.user,))
            conn.commit()
            st.success("Riwayat berhasil dihapus")
            st.rerun()

# ════════════════════════════════ HALAMAN DESKRIPSI ══════════════════════════════════
elif menu == "🪖 Deskripsi":
    st.title("🪖 Panduan Deteksi Helm")
    st.caption("Informasi kelas deteksi yang digunakan pada model")
    st.divider()

    kelas = [
        ("🔵 Pakai Helm", "#1a4a8a",
         "Pengendara terdeteksi menggunakan helm. Kondisi aman dan sesuai aturan lalu lintas yang berlaku. "
         "Kotak deteksi ditampilkan dengan warna BIRU."),
        ("🔴 Tanpa Helm", "#6a2020",
         "Pengendara terdeteksi tidak menggunakan helm. Berisiko tinggi dan melanggar aturan lalu lintas. "
         "Kotak deteksi ditampilkan dengan warna MERAH."),
    ]

    for nama, warna, deskripsi in kelas:
        st.markdown(f"""
        <div style="background:{warna}22;border-left:4px solid {warna}aa;
                    border-radius:10px;padding:1rem 1.2rem;margin-bottom:1rem;">
            <div style="font-size:1.1rem;font-weight:700;">{nama}</div>
            <div style="margin-top:0.4rem;">{deskripsi}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### ℹ️ Tentang Model")
    st.markdown("""
    <div style="background:#f0f0f011;border-radius:10px;padding:1rem 1.2rem;">
        <b>Model:</b> YOLOv11s<br>
        <b>Dataset:</b> Bike Helmet Detection Computer Vision Model <br>
        <b>Epochs:</b> 100<br>
        <b>mAP50:</b> ~ 0.889 <br>
        <b>Kelas:</b> With Helmet (🔵 Biru), Without Helmet (🔴 Merah)
    </div>
    """, unsafe_allow_html=True)
