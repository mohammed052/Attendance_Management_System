import streamlit as st
import pandas as pd
import face_recognition
import numpy as np
import cv2
from datetime import datetime

# Load student database
file_path = "Batch_A_data.csv"
attendance_file = "attendance_sheet.csv"
student_db = pd.read_csv(file_path)

# Convert string embeddings to numpy arrays if stored as strings
def convert_embedding_to_array(embedding_str):
    if isinstance(embedding_str, str):
        embedding_list = np.fromstring(embedding_str.strip("[]"), sep=", ")
        return embedding_list
    return np.array(embedding_str)

# Apply the conversion to all embeddings in the dataframe
student_db['encoding'] = student_db['encoding'].apply(convert_embedding_to_array)

MATCH_THRESHOLD = 0.6

# Function to match embedding with database
def match_student_euclidean_distance(face_embedding, student_db, threshold=MATCH_THRESHOLD):
    best_match_id = None
    best_match_name = None
    min_distance = float('inf')

    for _, student in student_db.iterrows():
        db_embedding = np.array(student['encoding'])
        if isinstance(db_embedding, np.ndarray) and face_embedding.shape == db_embedding.shape:
            distance = np.linalg.norm(face_embedding - db_embedding)
            if distance < min_distance and distance < threshold:
                min_distance = distance
                best_match_id = student['id']
                best_match_name = student['name']

    return best_match_id, best_match_name

# Function to process the image and match faces
def identify_students_in_image(image_path, student_db):
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    frame = cv2.imread(image_path)
    detected_student_ids = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        student_id, student_name = match_student_euclidean_distance(face_encoding, student_db)
        label = student_name.split()[0] if student_name else "Unknown"
        if student_id:
            detected_student_ids.append(student_id)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame, detected_student_ids

# Function to update attendance
def mark_attendance(detected_ids, attendance_file):
    today = datetime.now().strftime("%d-%m")
    if not detected_ids:
        return
    
    try:
        attendance = pd.read_csv(attendance_file)
    except FileNotFoundError:
        attendance = pd.DataFrame(columns=["id", "name", today])

    if today not in attendance.columns:
        attendance[today] = ""

    for student_id in detected_ids:
        if student_id in attendance["id"].values:
            attendance.loc[attendance["id"] == student_id, today] = "P"

    attendance.to_csv(attendance_file, index=False)

# Streamlit App
st.title("📸 Attendance Management System")
st.markdown(
    """
    Welcome to the Attendance Management System. Upload a photo to automatically detect students and mark attendance.
    """
)

uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    with st.spinner("Processing the image, please wait..."):
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        image_path = "temp_image.jpg"

        # Process the image
        try:
            processed_image, detected_ids = identify_students_in_image(image_path, student_db)
            mark_attendance(detected_ids, attendance_file)
            
            # Display results
            st.image(processed_image, caption="Processed Image with Detected Students", channels="BGR", use_container_width=True)
            if detected_ids:
                st.success(f"✅ Attendance marked for: {', '.join(map(str, detected_ids))}")
            else:
                st.warning("⚠️ No known students detected in the image.")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <small>Developed by Mohammed Bhadsorawala </small>
    """,
    unsafe_allow_html=True,
)