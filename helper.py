# helper.py
from ultralytics import YOLO
import streamlit as st
import cv2
from pytube import YouTube
import settings
import tempfile
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from models import DetectionHistory, SessionLocal

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def _display_detected_frames(conf, model, st_frame, image):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Predict the objects in the image using the YOLOv8 model
    res = model.predict(image, conf=conf)

    # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_youtube_video(conf, model):
    """
    Plays a YouTube video stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the YOLOv8 class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")

    if st.sidebar.button('Detect Objects'):
        try:
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the YOLOv8 class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH

    if 'detecting' not in st.session_state:
        st.session_state.detecting = False

    if st.sidebar.button('Detect Objects'):
        st.session_state.detecting = True

    if st.session_state.detecting:
        stop_button = st.sidebar.button('Stop')
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image)
                    if stop_button:
                        st.session_state.detecting = False
                        vid_cap.release()
                        break
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.session_state.detecting = False
            st.sidebar.error("Error loading video: " + str(e))

def play_stored_video(conf, model):
    """
    Plays a stored video file. Detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the YOLOv8 class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
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
                    _display_detected_frames(conf, model, st_frame, image)
                else:
                    vid_cap.release()
                    break
    else:
        st.warning("Please upload a video file.")



def save_detection(source_type, source_path, detected_image):
    """
    Save detection results to the database.
    """
    db = SessionLocal()
    new_record = DetectionHistory(
        source_type=source_type,
        source_path=source_path,
        detected_image=detected_image
    )
    db.add(new_record)
    db.commit()
    db.close()

def get_detection_history():
    """
    Retrieve detection history from the database.
    """
    db = SessionLocal()
    history = db.query(DetectionHistory).all()
    db.close()
    return history

def delete_detection_record(record_id):
    engine = create_engine(settings.DATABASE_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        record = session.query(DetectionHistory).get(record_id)
        if record:
            session.delete(record)
            session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()