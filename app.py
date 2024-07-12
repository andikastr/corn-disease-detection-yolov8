# Import modules
from pathlib import Path
import PIL
import tempfile
import cv2
import base64
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Corn Disease Detection using YOLOv8",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Customizing the sidebar with background color
st.markdown("""
    <style>
        .custom-header {
            background-color: #CB4A16;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .custom-header h2 {
            color: white;
        }
        .history-image {
            width: 80%;
            max-width: 300px;
            height: 50%;
        }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with custom header and different background color for the main content
st.sidebar.markdown('<div class="custom-header"><h2>🌽 | CornVerse</h2></div>', unsafe_allow_html=True)

page = st.sidebar.selectbox("Select Page", ["🏠 | Home", "🔎 | Detection", "⌛ | History"], index=0, key='page_selector')

# Home Page
if page == "🏠 | Home":
    st.title("Welcome to my CornVerse Project")
    st.write("""
        Harness the power of advanced technology with our web-based application utilizing YOLOv8 
        to accurately identify and diagnose diseases in corn leaves. Enhance your crop management with real-time, 
        reliable insights for healthier, more productive yields.
    """)
    st.subheader("Page Descriptions:")
    st.write("""
        - **Home**: Overview of the application and project description.
        - **Detection**: Upload an image and video or webcam to detect corn diseases using the YOLOv8 model.
        - **History**: View the history of previous detections, including the source type, source path, and detected images.
    """)

    col1, col2 = st.columns(2)

    with col1:
        default_image_path = str(settings.DEFAULT_IMAGE)
        default_image = PIL.Image.open(default_image_path)
        st.image(default_image_path, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
        default_detected_image = PIL.Image.open(default_detected_image_path)
        st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)


# Detection Page
elif page == "🔎 | Detection":
    st.title("Corn Disease Detection using YOLOv8")

    st.sidebar.header("ML Model Config")
    confidence = float(st.sidebar.slider("Select Model Confidence (%)", 25, 100, 40)) / 100

    model_path = Path(settings.DETECTION_MODEL)
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    st.sidebar.subheader("Image/Video Config")
    source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

    source_img = None
    source_vid = None

    # If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image", use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image", use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.")
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image', use_column_width=True)
            else:
                if st.sidebar.button('Detect Objects'):
                    try:
                        res = model.predict(uploaded_image, conf=confidence)
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]
                        st.image(res_plotted, caption='Detected Image', use_column_width=True)

                        # Save detection result
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
                            PIL.Image.fromarray(res_plotted).save(tmpfile.name)
                            with open(tmpfile.name, "rb") as file:
                                detected_image = file.read()
                                helper.save_detection("Image", source_img.name, detected_image)

                        try:
                            with st.expander("Detection Results"):
                                if boxes:
                                    for box in boxes:
                                        class_id = int(box.cls)
                                        class_name = model.names[class_id]
                                        conf_value = float(box.conf)
                                        st.write(f"Class: {class_name}, Confidence: {conf_value:.2f}")
                                else:
                                    st.write("No objects detected.")
                        except Exception as ex:
                            st.error("Error processing detection results.")
                            st.error(ex)
                    except Exception as ex:
                        st.error("Error running detection.")
                        st.error(ex)

    elif source_radio == settings.VIDEO:
        source_vid = st.sidebar.file_uploader("Choose a video...", type=("mp4", "avi", "mov", "mkv"))

        if source_vid is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(source_vid.read())
            vid_cap = cv2.VideoCapture(tfile.name)

            st.video(tfile.name)

            if st.sidebar.button('Detect Objects'):
                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        helper._display_detected_frames(confidence, model, st_frame, image)
                    else:
                        vid_cap.release()
                        break
        else:
            st.warning("Please upload a video file.")

    elif source_radio == settings.WEBCAM:
        helper.play_webcam(confidence, model)

    elif source_radio == settings.YOUTUBE:
        helper.play_youtube_video(confidence, model)

    else:
        st.error("Please select a valid source type!")

# History Page
elif page == "⌛ | History":
    st.title("Detection History")
    history = helper.get_detection_history()

    if not history:
        st.warning("No Detection History")
    else:
        for record in history:
            st.write(f"Source Type: {record.source_type}")
            st.write(f"Source Path: {record.source_path}")
            image_data = base64.b64encode(record.detected_image).decode('utf-8')
            st.markdown(f'<img class="history-image" src="data:image/png;base64,{image_data}" alt="Detected Image">', unsafe_allow_html=True)
            if st.button('Delete', key=f'delete_{record.id}'):
                helper.delete_detection_record(record.id)
                st.rerun()
