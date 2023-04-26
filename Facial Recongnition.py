import cv2

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Load the pre-trained face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')
# Initialize the video capture device
cap = cv2.VideoCapture(0)
while True:
    # Read a frame from the video capture device
    ret, frame = cap.read()
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # Loop over all detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Extract the face region from the grayscale frame
        roi_gray = gray[y:y + h, x:x + w]
        # Recognize the face using the pre-trained model
        id_, confidence = recognizer.predict(roi_gray)
        # Display the name of the recognized person if the confidence is high enough
        if confidence < 100:
            cv2.putText(frame, "Recognized person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('Facial Recognition', frame)
    # Exit the program when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()
