import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define landmark indices for right hand, elbow, and shoulder
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER
RIGHT_ELBOW = mp_pose.PoseLandmark.RIGHT_ELBOW
RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = (a[0] - b[0], a[1] - b[1])  # Vector BA
    c = (c[0] - b[0], c[1] - b[1])  # Vector BC
    dot_product = a[0] * c[0] + a[1] * c[1]
    magnitude_a = math.sqrt(a[0]**2 + a[1]**2)
    magnitude_c = math.sqrt(c[0]**2 + c[1]**2)
    angle = math.acos(dot_product / (magnitude_a * magnitude_c))
    return math.degrees(angle)

# Function to convert normalized coordinates to pixel coordinates
def get_real_coords(landmark, width, height):
    return int(landmark.x * width), int(landmark.y * height)

# Open the default camera
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Pose
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Get the dimensions of the frame
        height, width, _ = frame.shape

        # Convert the BGR image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Process the image and detect the pose
        results = pose.process(image)

        # Convert the image back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the pose annotation on the image
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmark coordinates
            landmarks = results.pose_landmarks.landmark

            # Get specific landmarks
            right_shoulder = get_real_coords(landmarks[RIGHT_SHOULDER], width, height)
            right_elbow = get_real_coords(landmarks[RIGHT_ELBOW], width, height)
            right_wrist = get_real_coords(landmarks[RIGHT_WRIST], width, height)

            # Calculate the angle
            angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Compare the angle with a given parameter (e.g., 45 degrees)
            angle_threshold = 45
            angle_status = "True" if abs(angle - angle_threshold) <= 10 else "False"

            color = (255, 0, 0)  # White color

            # Display the angle and status on the image at the top left corner
            text_y_offset = 30
            cv2.putText(image, f'Shoulder: {right_shoulder}', (10, text_y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            text_y_offset += 20
            cv2.putText(image, f'Elbow: {right_elbow}', (10, text_y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            text_y_offset += 20
            cv2.putText(image, f'Wrist: {right_wrist}', (10, text_y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            text_y_offset += 20
            cv2.putText(image, f'Angle: {angle:.2f} degrees', (10, text_y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
            text_y_offset += 20
            cv2.putText(image, f'Status: {angle_status}', (10, text_y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Pose Detection', image)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()