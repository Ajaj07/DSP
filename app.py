import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import time
import sys

# Page configuration
st.set_page_config(
    page_title="Image Processor Pro",
    page_icon="🖼️",
    layout="wide"
)

# Detect environment
def is_local_environment():
    """Detect if running locally or on cloud"""
    try:
        # Check if running on Streamlit Cloud
        if 'STREAMLIT_SHARING_MODE' in st.secrets or \
           'STREAMLIT_SERVER_HEADLESS' in st.secrets:
            return False
        
        # Try to access camera (will fail on cloud)
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            return True
        return False
    except:
        return False

# Initialize environment detection
if 'environment_checked' not in st.session_state:
    st.session_state.is_local = is_local_environment()
    st.session_state.environment_checked = True

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
    .success-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .env-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .env-local {
        background-color: #d4edda;
        color: #155724;
    }
    .env-cloud {
        background-color: #cce5ff;
        color: #004085;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🖼️ Image Processing Web App Pro</h1>', unsafe_allow_html=True)

# Environment badge
env_type = "🖥️ Local Mode" if st.session_state.is_local else "☁️ Cloud Mode"
env_class = "env-local" if st.session_state.is_local else "env-cloud"
st.markdown(f'<div style="text-align: center;"><span class="env-badge {env_class}">{env_type}</span></div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
Upload an image or use your camera for face detection with various processing options
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False
if 'last_frame' not in st.session_state:
    st.session_state.last_frame = None
if 'capture_count' not in st.session_state:
    st.session_state.capture_count = 0
if 'camera_image' not in st.session_state:
    st.session_state.camera_image = None

# Load face detection models
@st.cache_resource
def load_face_detectors():
    """Load all face detection models with error handling"""
    try:
        # Haar Cascades
        frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
        profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        
        # Validate cascades loaded correctly
        if frontal_cascade.empty() or alt_cascade.empty() or profile_cascade.empty():
            st.error("⚠️ Error loading Haar Cascade models")
            return None, None, None, None, False
        
        # DNN Face Detector (more accurate)
        try:
            modelFile = "opencv_face_detector_uint8.pb"
            configFile = "opencv_face_detector.pbtxt"
            net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
            has_dnn = True
        except Exception as e:
            net = None
            has_dnn = False
        
        return frontal_cascade, alt_cascade, profile_cascade, net, has_dnn
    except Exception as e:
        st.error(f"⚠️ Critical error loading face detection models: {str(e)}")
        return None, None, None, None, False

# Load models
models = load_face_detectors()
if models[0] is None:
    st.stop()

frontal_cascade, alt_cascade, profile_cascade, dnn_net, has_dnn_model = models

# Helper functions
def preprocess_for_detection(img):
    """Enhance image for better face detection"""
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
        gray = cv2.filter2D(gray, -1, kernel)
        
        return gray
    except Exception as e:
        st.error(f"Error in preprocessing: {str(e)}")
        return img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def detect_faces_dnn(img, confidence_threshold=0.5):
    """Detect faces using DNN (Deep Neural Network)"""
    try:
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        
        dnn_net.setInput(blob)
        detections = dnn_net.forward()
        
        faces = []
        img_with_faces = img.copy()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > confidence_threshold:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                faces.append((x1, y1, x2-x1, y2-y1))
                
                cv2.rectangle(img_with_faces, (x1, y1), (x2, y2), (0, 255, 0), 3)
                text = f'Face {int(confidence*100)}%'
                cv2.putText(img_with_faces, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return img_with_faces, len(faces)
    except Exception as e:
        st.error(f"Error in DNN face detection: {str(e)}")
        return img, 0

def detect_faces(img, sensitivity='medium', method='auto', confidence=0.5):
    """Detect faces in image with multiple methods"""
    try:
        if method == 'haar' or method == 'auto':
            enhanced = preprocess_for_detection(img)
            
            if sensitivity == 'high':
                scale_factor, min_neighbors, min_size = 1.05, 2, (15, 15)
            elif sensitivity == 'low':
                scale_factor, min_neighbors, min_size = 1.2, 7, (40, 40)
            else:
                scale_factor, min_neighbors, min_size = 1.1, 3, (20, 20)
            
            all_faces = []
            
            faces1 = frontal_cascade.detectMultiScale(
                enhanced, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                minSize=min_size, flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            faces2 = alt_cascade.detectMultiScale(
                enhanced, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
            )
            
            faces3 = profile_cascade.detectMultiScale(
                enhanced, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
            )
            
            all_faces = list(faces1) + list(faces2) + list(faces3)
            
            faces = []
            if len(all_faces) > 0:
                faces = cv2.groupRectangles(all_faces, groupThreshold=1, eps=0.2)[0]
            
            img_with_faces = img.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(img_with_faces, 'Face', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            haar_result = (img_with_faces, len(faces))
        
        if method == 'dnn' and has_dnn_model:
            return detect_faces_dnn(img, confidence)
        elif method == 'auto':
            if haar_result[1] > 0:
                return haar_result
            elif has_dnn_model:
                return detect_faces_dnn(img, confidence)
            else:
                return haar_result
        else:
            return haar_result if method == 'haar' else (img.copy(), 0)
    except Exception as e:
        st.error(f"Error in face detection: {str(e)}")
        return img, 0

def convert_to_grayscale(img):
    """Convert image to grayscale"""
    try:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        st.error(f"Error converting to grayscale: {str(e)}")
        return img

def colorize_grayscale(gray_img, colormap=cv2.COLORMAP_JET):
    """Apply pseudo-coloring to grayscale image"""
    try:
        return cv2.applyColorMap(gray_img, colormap)
    except Exception as e:
        st.error(f"Error colorizing image: {str(e)}")
        return gray_img

def pil_to_cv(pil_img):
    """Convert PIL image to OpenCV format"""
    try:
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        st.error(f"Error converting PIL to CV: {str(e)}")
        return np.array(pil_img)

def cv_to_pil(cv_img):
    """Convert OpenCV image to PIL format"""
    try:
        if len(cv_img.shape) == 2:
            return Image.fromarray(cv_img)
        else:
            return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    except Exception as e:
        st.error(f"Error converting CV to PIL: {str(e)}")
        return Image.fromarray(cv_img)

def create_download_link(img, filename="processed_image.png"):
    """Create download button for processed image"""
    try:
        pil_img = cv_to_pil(img)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        return buf
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return None

# Sidebar for controls
st.sidebar.header("⚙️ Controls")

# Mode selector - adaptive based on environment
if st.session_state.is_local:
    camera_modes = ["Upload Image", "Live Camera (Webcam)", "Camera Capture"]
    mode_help = "Choose between uploading an image, live webcam, or camera capture"
else:
    camera_modes = ["Upload Image", "Camera Capture"]
    mode_help = "Choose between uploading an image or using camera capture"

mode = st.sidebar.radio(
    "📷 Select Mode",
    camera_modes,
    help=mode_help
)

# Show environment info
if st.sidebar.checkbox("ℹ️ Show Environment Info", value=False):
    if st.session_state.is_local:
        st.sidebar.success("""
        **🖥️ Running Locally**
        - Live webcam available
        - Full camera access
        - Real-time processing
        """)
    else:
        st.sidebar.info("""
        **☁️ Running on Cloud**
        - Live webcam unavailable
        - Camera capture available
        - Upload images supported
        """)

# Common settings
sensitivity = "medium"
selected_colormap = None

if mode == "Upload Image":
    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG or PNG image"
    )
    
    # Operation selector
    operation = st.sidebar.selectbox(
        "Choose Operation",
        ["Original", "Face Detection", "Grayscale", "Colorize"],
        help="Select the image processing operation"
    )
    
    # Sensitivity slider for face detection
    if operation == "Face Detection":
        sensitivity = st.sidebar.select_slider(
            "Detection Sensitivity",
            options=["low", "medium", "high"],
            value="medium",
            help="Higher sensitivity detects more faces but may have false positives"
        )
        
        st.sidebar.markdown("""
        **💡 Tips for better detection:**
        - Use high sensitivity for old/low-quality photos
        - Ensure faces are clearly visible
        - Good lighting helps significantly
        """)
    
    # Colormap selector
    if operation == "Colorize":
        colormap_options = {
            "Jet (Rainbow)": cv2.COLORMAP_JET,
            "Hot": cv2.COLORMAP_HOT,
            "Cool": cv2.COLORMAP_COOL,
            "Spring": cv2.COLORMAP_SPRING,
            "Summer": cv2.COLORMAP_SUMMER,
            "Autumn": cv2.COLORMAP_AUTUMN,
            "Winter": cv2.COLORMAP_WINTER,
            "Ocean": cv2.COLORMAP_OCEAN,
            "Rainbow": cv2.COLORMAP_RAINBOW,
            "Viridis": cv2.COLORMAP_VIRIDIS
        }
        selected_colormap = st.sidebar.selectbox(
            "Choose Color Map",
            list(colormap_options.keys())
        )

elif mode == "Camera Capture":
    st.sidebar.markdown("### 📸 Camera Capture Settings")
    
    camera_sensitivity = st.sidebar.select_slider(
        "Detection Sensitivity",
        options=["low", "medium", "high"],
        value="medium",
        help="Adjust face detection sensitivity"
    )
    
    apply_detection = st.sidebar.checkbox(
        "Apply Face Detection",
        value=True,
        help="Automatically detect faces in captured image"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **💡 Camera Tips:**
    - Allow camera permissions
    - Ensure good lighting
    - Position yourself centrally
    - Click 'Take Photo' to capture
    """)

elif mode == "Live Camera (Webcam)":  # Only available locally
    st.sidebar.markdown("### 📹 Live Camera Settings")
    
    camera_sensitivity = st.sidebar.select_slider(
        "Detection Sensitivity",
        options=["low", "medium", "high"],
        value="medium",
        help="Adjust face detection sensitivity"
    )
    
    show_fps = st.sidebar.checkbox("Show FPS", value=True)
    
    camera_resolution = st.sidebar.selectbox(
        "Camera Resolution",
        ["640x480", "800x600", "1280x720"],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**💡 Camera Tips:**")
    st.sidebar.info("""
    - Ensure good lighting
    - Position your face centrally
    - Keep steady for best results
    - Click 'Capture Frame' to save
    """)

# Main content
if mode == "Upload Image":
    if uploaded_file is not None:
        try:
            # Read and convert image
            pil_image = Image.open(uploaded_file)
            cv_image = pil_to_cv(pil_image)
            
            # Display original image info
            st.sidebar.success("✅ Image uploaded successfully!")
            st.sidebar.info(f"**Dimensions:** {cv_image.shape[1]} x {cv_image.shape[0]}")
            
            # Create columns for side-by-side display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Original Image")
                st.image(pil_image, use_container_width=True)
            
            with col2:
                st.subheader(f"✨ Processed: {operation}")
                
                # Process image based on selected operation
                if operation == "Original":
                    processed_img = cv_image
                    st.image(cv_to_pil(processed_img), use_container_width=True)
                
                elif operation == "Face Detection":
                    with st.spinner("Detecting faces..."):
                        processed_img, face_count = detect_faces(cv_image, sensitivity)
                        st.image(cv_to_pil(processed_img), use_container_width=True)
                        
                        if face_count > 0:
                            st.success(f"🎯 **Faces Detected:** {face_count}")
                        else:
                            st.warning("⚠️ No faces detected in the image")
                            st.info("""
                            **Try these tips:**
                            - Increase sensitivity to 'high'
                            - Ensure faces are frontal and well-lit
                            - Check if image quality is sufficient
                            """)
                
                elif operation == "Grayscale":
                    processed_img = convert_to_grayscale(cv_image)
                    st.image(cv_to_pil(processed_img), use_container_width=True)
                    st.info("🎨 Image converted to grayscale")
                
                elif operation == "Colorize":
                    if len(cv_image.shape) == 3:
                        gray_img = convert_to_grayscale(cv_image)
                    else:
                        gray_img = cv_image
                    
                    processed_img = colorize_grayscale(gray_img, colormap_options[selected_colormap])
                    st.image(cv_to_pil(processed_img), use_container_width=True)
                    st.info(f"🌈 Pseudo-coloring applied: {selected_colormap}")
                
                # Download button
                if operation != "Original":
                    download_data = create_download_link(processed_img)
                    if download_data:
                        st.download_button(
                            label="⬇️ Download Processed Image",
                            data=download_data,
                            file_name=f"{operation.lower().replace(' ', '_')}_image.png",
                            mime="image/png"
                        )
        
        except Exception as e:
            st.error(f"⚠️ Error processing image: {str(e)}")
            st.info("Please try uploading a different image or check the file format.")
    
    else:
        # Show instructions when no image is uploaded
        st.info("👆 Please upload an image from the sidebar to get started!")
        
        st.markdown("""
        ### 🚀 How to Use:
        1. **Upload** an image using the file uploader in the sidebar
        2. **Select** an operation from the dropdown menu
        3. **View** the processed result side-by-side with the original
        4. **Download** the processed image using the download button
        
        ### 🔧 Available Operations:
        - **Face Detection**: Uses Haar Cascade classifier to detect and highlight faces
        - **Grayscale**: Converts color images to grayscale
        - **Colorize**: Applies various color maps to create pseudo-colored images
        """)

elif mode == "Camera Capture":
    st.markdown("### 📸 Camera Capture")
    
    # Use Streamlit's camera input
    camera_photo = st.camera_input("Take a photo")
    
    if camera_photo is not None:
        try:
            # Read the image
            pil_image = Image.open(camera_photo)
            cv_image = pil_to_cv(pil_image)
            
            st.success("✅ Photo captured successfully!")
            
            # Create columns for display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📷 Captured Photo")
                st.image(pil_image, use_container_width=True)
                st.info(f"**Dimensions:** {cv_image.shape[1]} x {cv_image.shape[0]}")
            
            with col2:
                if apply_detection:
                    st.subheader("🎯 Face Detection Result")
                    with st.spinner("Detecting faces..."):
                        processed_img, face_count = detect_faces(cv_image, camera_sensitivity)
                        st.image(cv_to_pil(processed_img), use_container_width=True)
                        
                        if face_count > 0:
                            st.success(f"✅ **Faces Detected:** {face_count}")
                        else:
                            st.warning("⚠️ No faces detected")
                            st.info("Try adjusting sensitivity or retaking the photo with better lighting")
                else:
                    st.subheader("📸 Original Photo")
                    st.image(pil_image, use_container_width=True)
                    processed_img = cv_image
            
            # Download button
            st.markdown("---")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                download_original = create_download_link(cv_image)
                if download_original:
                    st.download_button(
                        label="⬇️ Download Original",
                        data=download_original,
                        file_name="camera_capture_original.png",
                        mime="image/png",
                        use_container_width=True
                    )
            
            with col_dl2:
                if apply_detection:
                    download_processed = create_download_link(processed_img)
                    if download_processed:
                        st.download_button(
                            label="⬇️ Download with Detection",
                            data=download_processed,
                            file_name="camera_capture_detected.png",
                            mime="image/png",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"⚠️ Error processing camera image: {str(e)}")
    else:
        st.info("""
        ### 📸 Camera Capture Instructions:
        1. Click the **camera button** above to activate your camera
        2. Allow camera permissions in your browser
        3. Position yourself and click to take a photo
        4. Face detection will be applied automatically (if enabled)
        5. Download the original or processed image
        
        **Note:** This mode works on both local and cloud deployments!
        """)

elif mode == "Live Camera (Webcam)":  # Only available in local mode
    st.markdown("### 📹 Live Face Detection")
    
    # Camera controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        start_button = st.button("🎥 Start Camera", use_container_width=True)
    with col2:
        stop_button = st.button("⏹️ Stop Camera", use_container_width=True)
    with col3:
        capture_button = st.button("📸 Capture Frame", use_container_width=True)
    
    # Handle camera state
    if start_button:
        st.session_state.camera_active = True
    if stop_button:
        st.session_state.camera_active = False
    
    # Camera feed placeholder
    camera_placeholder = st.empty()
    info_placeholder = st.empty()
    
    if st.session_state.camera_active:
        try:
            # Parse resolution
            res_width, res_height = map(int, camera_resolution.split('x'))
            
            # Open camera
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
            
            if not cap.isOpened():
                st.error("⚠️ Cannot access camera. Please check camera permissions.")
                st.session_state.camera_active = False
            else:
                info_placeholder.success("✅ Camera is active! Processing live feed...")
                
                frame_count = 0
                start_time = time.time()
                
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    
                    if not ret:
                        st.error("⚠️ Failed to read from camera")
                        break
                    
                    # Detect faces in the frame
                    processed_frame, face_count = detect_faces(frame, camera_sensitivity)
                    
                    # Calculate FPS
                    frame_count += 1
                    if show_fps and frame_count % 10 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time
                        cv2.putText(processed_frame, f'FPS: {fps:.1f}', (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    # Display face count
                    cv2.putText(processed_frame, f'Faces: {face_count}', (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    # Store last frame for capture
                    st.session_state.last_frame = processed_frame.copy()
                    
                    # Display frame
                    camera_placeholder.image(cv_to_pil(processed_frame), 
                                           channels="RGB", 
                                           use_container_width=True)
                    
                    # Small delay to reduce CPU usage
                    time.sleep(0.03)
                
                cap.release()
                info_placeholder.info("📷 Camera stopped")
        
        except Exception as e:
            st.error(f"⚠️ Camera error: {str(e)}")
            st.info("Please ensure your camera is connected and you've granted browser permissions.")
            st.session_state.camera_active = False
    
    # Handle frame capture
    if capture_button and st.session_state.last_frame is not None:
        st.session_state.capture_count += 1
        st.success(f"📸 Frame captured! (Capture #{st.session_state.capture_count})")
        
        # Display captured frame
        st.subheader("Captured Frame")
        st.image(cv_to_pil(st.session_state.last_frame), use_container_width=True)
        
        # Download captured frame
        download_data = create_download_link(st.session_state.last_frame)
        if download_data:
            st.download_button(
                label="⬇️ Download Captured Frame",
                data=download_data,
                file_name=f"captured_frame_{st.session_state.capture_count}.png",
                mime="image/png",
                key=f"download_{st.session_state.capture_count}"
            )
    elif capture_button:
        st.warning("⚠️ No frame to capture. Start the camera first!")
    
    # Instructions
    if not st.session_state.camera_active and st.session_state.last_frame is None:
        st.info("""
        ### 🎥 Live Camera Instructions:
        1. Click **'Start Camera'** to begin live face detection
        2. Allow camera permissions in your browser
        3. Position yourself in front of the camera
        4. Click **'Capture Frame'** to save a snapshot
        5. Click **'Stop Camera'** when finished
        
        **Note:** Live detection runs in real-time with face highlighting and counting.
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit & OpenCV")
st.sidebar.caption(f"Session Captures: {st.session_state.capture_count}")

# Add deployment tips at the bottom
with st.expander("🚀 Deployment Information"):
    st.markdown("""
    ### Environment Detection
    This app automatically detects whether it's running locally or on Streamlit Cloud and adapts accordingly:
    
    **🖥️ Local Environment:**
    - Full live webcam support with real-time face detection
    - Camera capture mode available
    - Image upload functionality
    
    **☁️ Cloud Environment (Streamlit Cloud):**
    - Camera capture mode (single photo)
    - Image upload functionality
    - Live webcam disabled (not supported on cloud)
    
    ### Requirements
    Your `requirements.txt` is perfect as-is:
    ```
    streamlit
    opencv-python-headless
    numpy
    Pillow
    ```
    
    ### Running Locally
    ```bash
    streamlit run app.py
    ```
    
    ### Deploying to Streamlit Cloud
    1. Push your code to GitHub
    2. Connect to Streamlit Cloud
    3. The app will automatically detect cloud environment
    4. Camera capture mode will work seamlessly
    
    **Note:** The app intelligently switches features based on environment!
    """)
