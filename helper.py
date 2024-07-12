from ultralytics import YOLO
import streamlit as st
import cv2
import yt_dlp
import settings
import tempfile
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
import av
import numpy as np
from database import DetectionHistory, SessionLocal

def load_model(model_path):

    model = YOLO(model_path)
    return model

def _display_detected_frames(conf, model, st_frame, image):
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

class VideoProcessor(VideoProcessorBase):

    def __init__(self, confidence, model):
        self.confidence = confidence
        self.model = model

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Predict the objects in the image using the YOLOv8 model
        res = self.model.predict(image, conf=self.confidence)

        # Plot the detected objects on the video frame
        res_plotted = res[0].plot()
        
        return av.VideoFrame.from_ndarray(res_plotted, format="bgr24")

def play_webcam(conf, model):

    webrtc_ctx = webrtc_streamer(
        key="webcam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=lambda: VideoProcessor(conf, model),
        media_stream_constraints={"video": True, "audio": False},
    )

    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.confidence = conf
        webrtc_ctx.video_processor.model = model

def play_youtube_video(conf, model):

    source_youtube = st.sidebar.text_input("YouTube Video url")

    if st.sidebar.button('Detect Objects'):
        try:
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': '%(id)s.%(ext)s',
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source_youtube, download=False)
                url = info['url']
            
            vid_cap = cv2.VideoCapture(url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, model, st_frame, image)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_stored_video(conf, model):
    
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
