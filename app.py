import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Image Processor",
    page_icon="🖼️",
    layout="wide"
)

# Title and description
st.title("🖼️ Image Processing Web App")
st.markdown("""
Upload an image and choose from various processing options:
- **Face Detection**: Detect and count faces in the image
- **Grayscale**: Convert color image to grayscale
- **Colorize**: Apply pseudo-coloring to grayscale images
""")

# Load face detection models
@st.cache_resource
def load_face_detectors():
    # Haar Cascades
    frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    # DNN Face Detector (more accurate)
    try:
        modelFile = "opencv_face_detector_uint8.pb"
        configFile = "opencv_face_detector.pbtxt"
        
        # Try to load DNN model
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
        has_dnn = True
    except:
        net = None
        has_dnn = False
    
    return frontal_cascade, alt_cascade, profile_cascade, net, has_dnn

frontal_cascade, alt_cascade, profile_cascade, dnn_net, has_dnn_model = load_face_detectors()

# Helper functions
def preprocess_for_detection(img):
    """Enhance image for better face detection"""
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    return gray

def detect_faces_dnn(img, confidence_threshold=0.5):
    """Detect faces using DNN (Deep Neural Network) - more accurate"""
    h, w = img.shape[:2]
    
    # Create blob from image
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
    
    # Set input and perform detection
    dnn_net.setInput(blob)
    detections = dnn_net.forward()
    
    faces = []
    img_with_faces = img.copy()
    
    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            faces.append((x1, y1, x2-x1, y2-y1))
            
            # Draw rectangle and confidence
            cv2.rectangle(img_with_faces, (x1, y1), (x2, y2), (0, 255, 0), 3)
            text = f'Face {int(confidence*100)}%'
            cv2.putText(img_with_faces, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_with_faces, len(faces)

def detect_faces(img, sensitivity='medium', method='auto', confidence=0.5):
    """Detect faces in image with multiple methods"""
    
    # Enhanced preprocessing
    if method == 'haar' or method == 'auto':
        enhanced = preprocess_for_detection(img)
        
        # Set parameters based on sensitivity
        if sensitivity == 'high':
            scale_factor = 1.05
            min_neighbors = 2
            min_size = (15, 15)
        elif sensitivity == 'low':
            scale_factor = 1.2
            min_neighbors = 7
            min_size = (40, 40)
        else:  # medium
            scale_factor = 1.1
            min_neighbors = 3
            min_size = (20, 20)
        
        # Try multiple cascades
        all_faces = []
        
        # Frontal face detection (primary)
        faces1 = frontal_cascade.detectMultiScale(
            enhanced, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors, 
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Alternative frontal face detection
        faces2 = alt_cascade.detectMultiScale(
            enhanced, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors, 
            minSize=min_size
        )
        
        # Profile face detection
        faces3 = profile_cascade.detectMultiScale(
            enhanced, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors, 
            minSize=min_size
        )
        
        # Combine all detections
        all_faces = list(faces1) + list(faces2) + list(faces3)
        
        # Remove duplicate detections
        faces = []
        if len(all_faces) > 0:
            faces = cv2.groupRectangles(all_faces, groupThreshold=1, eps=0.2)[0]
        
        # Draw rectangles
        img_with_faces = img.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img_with_faces, 'Face', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        haar_result = (img_with_faces, len(faces))
    
    # Try DNN method
    if method == 'dnn' and has_dnn_model:
        return detect_faces_dnn(img, confidence)
    elif method == 'auto':
        # If Haar found faces, return that
        if haar_result[1] > 0:
            return haar_result
        # Otherwise try DNN if available
        elif has_dnn_model:
            return detect_faces_dnn(img, confidence)
        else:
            return haar_result
    else:
        return haar_result if method == 'haar' else (img.copy(), 0)

def convert_to_grayscale(img):
    """Convert image to grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def colorize_grayscale(gray_img, colormap=cv2.COLORMAP_JET):
    """Apply pseudo-coloring to grayscale image"""
    return cv2.applyColorMap(gray_img, colormap)

def pil_to_cv(pil_img):
    """Convert PIL image to OpenCV format"""
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def cv_to_pil(cv_img):
    """Convert OpenCV image to PIL format"""
    if len(cv_img.shape) == 2:  # Grayscale
        return Image.fromarray(cv_img)
    else:  # Color
        return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def create_download_link(img, filename="processed_image.png"):
    """Create download button for processed image"""
    pil_img = cv_to_pil(img)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf

# Sidebar for controls
st.sidebar.header("⚙️ Controls")

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
else:
    sensitivity = "medium"

# Colormap selector (only shown for Colorize operation)
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

# Main content
if uploaded_file is not None:
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
                - Try preprocessing the image first
                """)
        
        elif operation == "Grayscale":
            processed_img = convert_to_grayscale(cv_image)
            st.image(cv_to_pil(processed_img), use_container_width=True)
            st.info("🎨 Image converted to grayscale")
        
        elif operation == "Colorize":
            # First convert to grayscale if needed
            if len(cv_image.shape) == 3:
                gray_img = convert_to_grayscale(cv_image)
            else:
                gray_img = cv_image
            
            # Apply colormap
            processed_img = colorize_grayscale(gray_img, colormap_options[selected_colormap])
            st.image(cv_to_pil(processed_img), use_container_width=True)
            st.info(f"🌈 Pseudo-coloring applied: {selected_colormap}")
        
        # Download button
        if operation != "Original":
            st.download_button(
                label="⬇️ Download Processed Image",
                data=create_download_link(processed_img),
                file_name=f"{operation.lower().replace(' ', '_')}_image.png",
                mime="image/png"
            )

else:
    # Show instructions when no image is uploaded
    st.info("👆 Please upload an image from the sidebar to get started!")
    
    # Display example usage
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
    
    ### 💡 Tips:
    - Face detection works best with clear, frontal face images
    - Try different color maps for creative effects
    - All processed images can be downloaded in PNG format
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ using Streamlit & OpenCV")