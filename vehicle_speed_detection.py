import cv2
import numpy as np
import os
import math

def calculate_speed(prev_pos, curr_pos, fps, conversion_factor=3.6):
    """
    Calculate the speed of an object in km/h.
    """
    distance = math.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)
    speed = (distance * fps) * conversion_factor / 100  # Assuming 100 pixels = 1 meter
    return speed

def detect_speed(video_path, speed_limit=50, output_folder="Penalties"):
    """
    Detect and estimate the speed of objects from a video.
    Capture and store images of vehicles exceeding the speed limit.
    """
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video file '{video_path}' not found!")

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_count = 0
    penalty_count = 0  # Count of penalty vehicles

    # Object tracking data
    object_positions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        height, width, channels = frame.shape

        # Simplified detection: detect motion using contours
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if frame_count == 1:
            prev_frame = blurred
            continue

        # Compute frame difference
        frame_delta = cv2.absdiff(prev_frame, blurred)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Find contours (motion detection)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_object_positions = {}

        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 500:  # Filter small objects
                continue

            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            new_object_positions[i] = (center_x, center_y)

            # Draw bounding box (for reference)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Calculate and display speed
        for obj_id, curr_pos in new_object_positions.items():
            if obj_id in object_positions:
                prev_pos = object_positions[obj_id]
                speed = calculate_speed(prev_pos, curr_pos, fps)

                # Display speed on the frame (for reference)
                cv2.putText(frame, f"Speed: {speed:.2f} km/h", (curr_pos[0], curr_pos[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Check for speed limit violation
                if speed > speed_limit:
                    # Display penalty warning (for reference)
                    cv2.putText(frame, "Speed Limit Exceeded!", (curr_pos[0], curr_pos[1] - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, "PENALTY!", (curr_pos[0], curr_pos[1] - 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # Save penalty image
                    penalty_count += 1
                    penalty_image_path = os.path.join(output_folder, f"Penalty_{penalty_count}.jpg")
                    vehicle_image = frame[y:y + 100, x:x + 100]  # Crop vehicle region
                    cv2.imwrite(penalty_image_path, vehicle_image)

        object_positions = new_object_positions
        prev_frame = blurred

        # Display the frame with annotations
        cv2.imshow("Speed Detection", frame)

        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Penalty images saved in folder: {output_folder}")

# Example usage
video_path = r"C:/Users\ANAND S\Downloads\Vehicle Speed Detection project\test video 2.mp4"  # Update path
speed_limit = 50  # Set the speed limit in km/h
detect_speed(video_path, speed_limit)
