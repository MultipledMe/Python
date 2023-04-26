import cv2
import numpy as np

# Load the Eyelike model
eyelike_model = cv2.face.createFacemarkLBF()
eyelike_model.loadModel('lbfmodel.yaml')

# Create a VideoCapture object to capture video from the default camera
cap = cv2.VideoCapture(0)

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = cv2.face.detectMultiScale(gray)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the grayscale image
        face_gray = gray[y:y + h, x:x + w]

        # Detect eyes in the face region using the Eyelike model
        _, landmarks = eyelike_model.fit(frame, faces)
        left_eye = landmarks[0][36:42]
        right_eye = landmarks[0][42:48]

        # Draw a circle around each eye
        for (x_eye, y_eye) in left_eye:
            cv2.circle(frame, (x + x_eye, y + y_eye), 2, (0, 255, 0), -1)
        for (x_eye, y_eye) in right_eye:
            cv2.circle(frame, (x + x_eye, y + y_eye), 2, (0, 255, 0), -1)

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
