# import cv2

# # RTSP URL of the stream
# rtsp_url = 'rtsp://localhost:8554/mystream'

# # Create a VideoCapture object
# cap = cv2.VideoCapture(rtsp_url)

# # Check if the camera opened successfully
# if not cap.isOpened():
#     print("Error: Unable to open RTSP stream.")
#     exit(1)

# # Read frames from the stream
# #while True:
# ret, frame = cap.read()
# if not ret:
#     print("Error: Unable to read frame.")
#     exit(1)

# # Display the frame
# cv2.imshow('Frame', frame)

# # Break the loop when 'q' is pressed
# if cv2.waitKey() & 0xFF == ord('q'):
    

# # Release the VideoCapture object and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()

import os
import cv2

rtsp_url = os.environ['RTSP_URL']

# Create a VideoCapture object
cap = cv2.VideoCapture(rtsp_url)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open RTSP stream.")
    exit()
    
# Read a single frame from the stream
ret, frame = cap.read()

# Check if the frame was read successfully
if ret:
    # Display the frame
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)  # Wait indefinitely until any key is pressed
else:
    print("Error: Unable to read frame.")

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
