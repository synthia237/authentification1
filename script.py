import os
import cv2
import face_recognition


def load_known_images():
        
        known_encodings = []
        known_names = []

        # Path to the training images directory
        training_images_dir = os.path.join('media')

        # Loop through each subdirectory in the training images directory
        for person_dir in os.listdir(training_images_dir):
            person_name = person_dir

            # Loop through each image in the person's subdirectory
            for image_file in os.listdir(os.path.join(training_images_dir, person_dir)):
                image_path = os.path.join(training_images_dir, person_dir, image_file)

                # Load the image and compute the face encoding
                image = face_recognition.load_image_file(image_path)
                encoding = face_recognition.face_encodings(image)[0]

                # Append the encoding and name to the lists
                known_encodings.append(encoding)
                known_names.append(person_name)

        return known_encodings, known_names

    # Load the known images and encodings
known_encodings, known_names = load_known_images()

    # Initialize the webcam
video_capture = cv2.VideoCapture(0)

    # Load the Haar cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
        # Capture frame-by-frame from the webcam
        ret, frame = video_capture.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using the Haar cascade classifier
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Loop through each face found in the frame
        for (x, y, w, h) in faces:
            # Crop the face region from the frame
            face_image = frame[y:y+h, x:x+w]

            # Resize the face image for better recognition performance
            face_image = cv2.resize(face_image, (0, 0), fx=0.25, fy=0.25)

            # Find the face encodings in the resized face image
            face_encodings = face_recognition.face_encodings(face_image)

            # Loop through each face encoding found in the resized face image
            for face_encoding in face_encodings:
                # Compare the face encoding with the known encodings
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                name = "Unknown"

                # If there is a match, find the index in known_encodings
                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]

                # Draw a rectangle around the face and display the name
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        """
        cv2.imshow('Real-time Facial Recognition', frame)
"""
        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()