import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# --- IMPORTS FOR REAL-TIME VIDEO STREAMING ---
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
# --------------------------------------------------

# Page configuration
st.set_page_config(
    page_title="Image Processor Pro",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🖼️ Image Processing Web App Pro</h1>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
Upload an image or use your camera for face detection with various processing options.
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'webrtc_face_count' not in st.session_state:
    st.session_state.webrtc_face_count = 0

# Load face detection models
@st.cache_resource
def load_face_detectors():
    """Load all face detection models with error handling"""
    try:
        frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if frontal_cascade.empty():
            st.error("⚠️ Error loading frontal face Haar Cascade model. Please check your OpenCV installation.")
            return None
        return frontal_cascade
    except Exception as e:
        st.error(f"⚠️ Critical error loading face detection models: {str(e)}")
        return None

frontal_cascade = load_face_detectors()

# Helper functions
def preprocess_for_detection(img):
    """Enhance image for better face detection"""
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        # Apply histogram equalization to improve contrast
        gray = cv2.equalizeHist(gray)
        return gray
    except Exception:
        # Fallback if preprocessing fails
        return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def detect_faces(img, sensitivity='medium'):
    """Detect faces in image using Haar Cascades"""
    if frontal_cascade is None:
        return img, 0
    try:
        # Preprocess the image for better detection
        enhanced = preprocess_for_detection(img)
            
        # Adjust detection parameters based on sensitivity
        if sensitivity == 'high':
            scale_factor, min_neighbors, min_size = 1.05, 3, (20, 20)
        elif sensitivity == 'low':
            scale_factor, min_neighbors, min_size = 1.2, 7, (50, 50)
        else: # medium
            scale_factor, min_neighbors, min_size = 1.1, 5, (30, 30)
            
        faces = frontal_cascade.detectMultiScale(
            enhanced, scaleFactor=scale_factor, minNeighbors=min_neighbors,
            minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE
        )
            
        img_with_faces = img.copy()
        for (x, y, w, h) in faces:
            # Draw a green rectangle around the face
            cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 3)
            # Add a label
            cv2.putText(img_with_faces, 'Face', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return (img_with_faces, len(faces))
    except Exception:
        # If any error occurs during detection, return the original image
        return img, 0

def convert_to_grayscale(img):
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception:
        return img

def colorize_grayscale(gray_img, colormap=cv2.COLORMAP_JET):
    try:
        return cv2.applyColorMap(gray_img, colormap)
    except Exception:
        return gray_img

def pil_to_cv(pil_img):
    try:
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        return np.array(pil_img)

def cv_to_pil(cv_img):
    try:
        if len(cv_img.shape) == 2: # Grayscale
            return Image.fromarray(cv_img)
        else: # Color
            return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    except Exception:
        return Image.fromarray(cv_img)

def create_download_link(img, filename="processed_image.png"):
    try:
        pil_img = cv_to_pil(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return None

# --- UPDATED Video Processor Class for streamlit-webrtc ---
# This uses the modern and recommended `VideoProcessorBase` class
class RealTimeFaceDetector(VideoProcessorBase):
    """A class to handle real-time frame processing."""
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        # Dynamically get sensitivity from the session state, which is set by the sidebar slider
        sensitivity = st.session_state.get("webrtc_sensitivity", "medium")
        
        processed_img, face_count = detect_faces(img, sensitivity=sensitivity)
        
        st.session_state.webrtc_face_count = face_count

        # Add face count text onto the processed image
        cv2.putText(processed_img, f'Faces: {face_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")

mode = st.sidebar.radio(
    "📷 Select Mode",
    ["Upload Image", "Live Camera (Webcam)", "Camera Capture"],
    help="Choose your image source and processing mode."
)

# --- Main Content Area ---
if mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    operation = st.sidebar.selectbox("Choose Operation", ["Face Detection", "Grayscale", "Colorize"])
    sensitivity = "medium"

    if operation == "Face Detection":
        sensitivity = st.sidebar.select_slider("Detection Sensitivity", options=["low", "medium", "high"], value="medium")
    
    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file)
        cv_image = pil_to_cv(pil_image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original Image")
            st.image(pil_image, use_container_width=True)
        
        with col2:
            st.subheader(f"✨ Processed: {operation}")
            processed_img = cv_image
            
            if operation == "Face Detection":
                with st.spinner("Detecting faces..."):
                    processed_img, face_count = detect_faces(cv_image, sensitivity)
                    st.image(cv_to_pil(processed_img), use_container_width=True)
                    if face_count > 0:
                        st.success(f"🎯 Found {face_count} face(s)!")
                    else:
                        st.warning("⚠️ No faces detected.")
            
            elif operation == "Grayscale":
                processed_img = convert_to_grayscale(cv_image)
                st.image(cv_to_pil(processed_img), use_container_width=True)
            
            elif operation == "Colorize":
                gray_img = convert_to_grayscale(cv_image)
                processed_img = colorize_grayscale(gray_img)
                st.image(cv_to_pil(processed_img), use_container_width=True)
            
            download_data = create_download_link(processed_img)
            if download_data:
                st.download_button(
                    label="⬇️ Download Processed Image",
                    data=download_data,
                    file_name=f"processed_{uploaded_file.name}",
                    mime="image/png"
                )
    else:
        st.info("👆 Please upload an image using the sidebar to begin.")

elif mode == "Camera Capture":
    st.subheader("📸 Take a Snapshot")
    camera_photo = st.camera_input("Position yourself and click the button to take a photo.")
    apply_detection = st.sidebar.checkbox("Apply Face Detection", value=True)
    
    if apply_detection:
        sensitivity = st.sidebar.select_slider("Detection Sensitivity", options=["low", "medium", "high"], value="medium")

    if camera_photo is not None:
        pil_image = Image.open(camera_photo)
        cv_image = pil_to_cv(pil_image)
        
        if apply_detection:
            st.subheader("🎯 Face Detection Result")
            with st.spinner("Detecting faces..."):
                processed_img, face_count = detect_faces(cv_image, sensitivity)
                st.image(cv_to_pil(processed_img), use_container_width=True)
                if face_count > 0:
                    st.success(f"🎯 Found {face_count} face(s)!")
                else:
                    st.warning("⚠️ No faces detected.")
                
                download_data = create_download_link(processed_img)
                if download_data:
                    st.download_button("⬇️ Download Processed Photo", download_data, "snapshot_processed.png", "image/png")
        else:
            st.subheader("📷 Your Snapshot")
            st.image(pil_image, use_container_width=True)
            download_data = create_download_link(cv_image)
            if download_data:
                st.download_button("⬇️ Download Photo", download_data, "snapshot.png", "image/png")

elif mode == "Live Camera (Webcam)":
    st.subheader("📹 Live Face Detection")
    st.sidebar.info("Adjust sensitivity for the live stream. A stable internet connection is recommended.")
    
    # Add a key to the slider to access its value from the VideoProcessorBase class
    st.sidebar.select_slider(
        "Detection Sensitivity",
        options=["low", "medium", "high"],
        value="medium",
        key="webrtc_sensitivity"
    )

    ctx = webrtc_streamer(
        key="face-detection-stream",
        video_processor_factory=RealTimeFaceDetector,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        sendback_audio=False
    )

    if ctx.state.playing:
        st.success(f"✅ Live stream active! **Faces detected: {st.session_state.get('webrtc_face_count', 0)}**")
    else:
        st.info("Click 'START' to activate your camera and begin real-time face detection.")

# Footer in the sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit & OpenCV")
