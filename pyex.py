import streamlit as st
import cv2
from ultralytics import YOLO
import winsound

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Autonomous Accident Prevention System",
    layout="wide"
)

# ---------------- SESSION STATE ----------------
if "run" not in st.session_state:
    st.session_state.run = False

# ---------------- TITLE ----------------
st.title("🏎️ Autonomous Accident Prevention System (AAPS)")

st.markdown("""
            
### 🔐 About the System
An **Autonomous Accident Prevention System (ADAS)** is an AI-powered safety system
designed to **detect obstacles**, **warn the driver**, and **simulate automatic braking**
to reduce road accidents.

This demo uses:
- **YOLOv8** for object detection
- **Camera-based distance estimation**
- **Real-time visual + sound alerts**
""")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ System Controls")

alert_enable = st.sidebar.checkbox("Enable Sound Alert", True)

camera_index = st.sidebar.selectbox(
    "Select Camera",
    [0, 1],
    index=0
)

st.sidebar.info("""
### Alert Logic
🟢 SAFE – Object far  
❗ WARNING – Object close  
⚠️ DANGER – Object very close  
➡️ AUTO BRAKE simulated
""")

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ Start AAPA"):
        st.session_state.run = True

with col2:
    if st.button("⏹️ Stop AAPA"):
        st.session_state.run = False

# ---------------- LOAD MODEL ----------------
model = YOLO("yolov8n.pt")

frame_placeholder = st.empty()
status_placeholder = st.empty()

# ---------------- CAMERA LOOP ----------------
if st.session_state.run:

    cap = cv2.VideoCapture(camera_index)
    status_placeholder.success("✅ ADAS Camera Started")

    while st.session_state.run and cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("❌ Camera Error")
            break

        results = model(frame, stream=True)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                area = w * h  # Distance approximation

                # ---------- ADAS LOGIC ----------
                if area > 80000:
                    color = (0, 0, 255)
                    label = "DANGER - AUTO BRAKE"

                    if alert_enable:
                        winsound.Beep(1000, 300)

                    cv2.putText(frame, "AUTO BRAKE ACTIVATED",
                                (40, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.1, (0, 0, 255), 3)

                elif area > 30000:
                    color = (0, 255, 255)
                    label = "WARNING"

                    if alert_enable:
                        winsound.Beep(700, 200)

                else:
                    color = (0, 255, 0)
                    label = "SAFE"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()
    status_placeholder.info("🛑 ADAS Stopped")

# ---------------- INFORMATION SECTION ----------------
st.markdown("---")
st.subheader("🧠 How Accident Prevention Works")

st.markdown("""
1. Camera captures live road video  
2. YOLOv8 detects nearby objects  
3. Bounding box size estimates distance  
4. System classifies risk level  
5. Warning or Auto-Brake is triggered  
""")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
### 🤖 Future Enhancements
• Speed limit detection  
• Real braking hardware integration  
• Vehicle number recognition  
• Driver fatigue detection  
• Cloud traffic analytics  

⚠️ **Educational Prototype – Not for real vehicle deployment**
""")


