import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import math
import time
from datetime import datetime
from pygame import mixer
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading
import geocoder  # To get live location

# Initialize pygame mixer
mixer.init()


# Email credentials and configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = "adarshayush02345@gmail.com"  # Replace with your email
EMAIL_PASS = "lohj qdlc pjgp nasv"        # Replace with your email password
TO_EMAIL = "raviraj18012000@gmail.com"  # Replace with recipient's email



# Function to get live location (using IP address)
def get_live_location():
    g = geocoder.ip('me')  # Fetch the location based on your IP
    if g.latlng:
        latitude, longitude = g.latlng
        return f"Latitude: {latitude}, Longitude: {longitude}"
    else:
        return "Location not available."


# Function to send email with specific animal name, location, and time
def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_USER
        msg["To"] = TO_EMAIL
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")


# Alarm sound function using pygame
def play_alarm():
    mixer.music.load('alarm.mp3')  # Replace with the path to your alarm file
    mixer.music.play()


# Load YOLO model (replace 'best.pt' with your actual model path)
model = YOLO('best.pt')

# All wild animal names (extended list)
classnames = [
    'antelope', 'bear', 'cheetah', 'human', 'coyote', 'crocodile', 'deer', 'elephant', 'flamingo',
    'fox', 'giraffe', 'gorilla', 'hedgehog', 'hippopotamus', 'hornbill', 'horse', 'hummingbird', 'hyena',
    'kangaroo', 'koala', 'leopard', 'lion', 'meerkat', 'mole', 'monkey', 'moose', 'okapi', 'orangutan',
    'ostrich', 'otter', 'panda', 'pelecaniformes', 'porcupine', 'raccoon', 'reindeer', 'rhino', 'rhinoceros',
    'snake', 'squirrel', 'swan', 'tiger', 'turkey', 'wolf', 'woodpecker', 'zebra','horse',
    'wild boar', 'bison', 'buffalo', 'panther', 'jaguar', 'leopard', 'puma', 'cheetah', 'elephant seal',
    'grizzly bear', 'polar bear', 'giant panda', 'red panda', 'chimpanzee', 'orangutan', 'snow leopard',
    'sea lion', 'walrus', 'manatee', 'hippopotamus', 'wolverine', 'wild dog', 'serval', 'jackal',
    'meerkat', 'hyena', 'camel', 'moose', 'albatross', 'vulture', 'eagle', 'falcon', 'owl', 'hawk', 'kite',
    'penguin', 'seal', 'whale', 'orca', 'dolphin', 'shark', 'ray', 'stingray', 'cuttlefish', 'octopus'
]

# Wild animal classes
wild_animals = set(classnames)

# Human detection class
human_class = "human"

# Streamlit App Interface
st.title("Animal Detection System")
st.sidebar.title("Options")

option = st.sidebar.selectbox("Choose an option:", ("Upload an Image", "Upload a Video", "Open Webcam"))

# Get live location (automatically)
location = get_live_location()
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Detect human and animal
def detect_human_and_animal(frame):
    result = model(frame, stream=True)
    human_detected = False
    detected_humans = []
    wild_animal_detected = False
    detected_animals = []

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            class_index = int(box.cls[0])

            # Detect human
            if confidence > 50 and classnames[class_index] == human_class:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f'Human {confidence}%', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                human_detected = True
                detected_humans.append("Human")

            # Detect wild animals
            elif confidence > 50 and classnames[class_index] in wild_animals:
                detected_animal = classnames[class_index]
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'{detected_animal} {confidence}%', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                wild_animal_detected = True
                detected_animals.append(detected_animal)

    return frame, human_detected, detected_humans, wild_animal_detected, detected_animals


if option == "Upload an Image":
    uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        frame, human_detected, detected_humans, wild_animal_detected, detected_animals = detect_human_and_animal(image)

        st.image(frame, channels="BGR")
        if human_detected:
            st.success("Human detected in the uploaded image!")
        if wild_animal_detected:
            for animal in detected_animals:
                threading.Thread(target=play_alarm).start()
                threading.Thread(target=send_email, args=(
                    f"Wild Animal Detected: {animal}",
                    f"A wild animal, specifically a {animal}, was detected in the uploaded image at {current_time}.\nLocation: {location}"
                )).start()
            st.warning(f"Warning: {', '.join(detected_animals)} Detected!")

if option == "Open Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to open webcam. Check permissions!")
            break

        frame, human_detected, detected_humans, wild_animal_detected, detected_animals = detect_human_and_animal(frame)

        stframe.image(frame, channels="BGR")

        if human_detected:
            st.success("Human detected via webcam!")
        if wild_animal_detected:
            for animal in detected_animals:
                threading.Thread(target=play_alarm).start()
                threading.Thread(target=send_email, args=(
                    f"Wild Animal Detected: {animal}",
                    f"A wild animal, specifically a {animal}, was detected via the webcam at {current_time}.\nLocation: {location}"
                )).start()

if option == "Upload a Video":
    uploaded_file = st.sidebar.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        vid_cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while vid_cap.isOpened():
            ret, frame = vid_cap.read()
            if not ret:
                break

            frame, human_detected, detected_humans, wild_animal_detected, detected_animals = detect_human_and_animal(frame)

            current_time = vid_cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # Current time of the video in seconds

            stframe.image(frame, channels="BGR")

            if human_detected:
                st.success("Human detected in the video!")
            if wild_animal_detected:
                for animal in detected_animals:
                    threading.Thread(target=play_alarm).start()
                    threading.Thread(target=send_email, args=(
                        f"Wild Animal Detected: {animal}",
                        f"A wild animal, specifically a {animal}, was detected in the video at {int(current_time // 60)} min {int(current_time % 60)} sec!\nLocation: {location}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )).start()
                st.warning(f"Warning: {', '.join(detected_animals)} Detected at {int(current_time // 60)} min {int(current_time % 60)} sec!")
