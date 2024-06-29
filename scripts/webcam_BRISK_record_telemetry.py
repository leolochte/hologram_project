import cv2
import numpy as np
import time

# Global variables to store keypoints and tracked keypoint
keypoints = []
tracked_keypoint = None
prev_gray = None

# Function to detect and draw keypoints
def detect_keypoints(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize BRISK detector
    brisk = cv2.BRISK_create()
    
    # Detect keypoints
    global keypoints
    start_time = time.time()
    keypoints, _ = brisk.detectAndCompute(gray, None)
    feature_extraction_time = (time.time() - start_time) * 1000  # Convert to milliseconds
    
    # Draw keypoints as red dots
    for kp in keypoints:
        x, y = kp.pt
        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1) # Red color
        
    return frame, gray, feature_extraction_time

# Mouse click event handler
def mouse_click(event, x, y, flags, param):
    global keypoints, tracked_keypoint
    
    # On left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        closest_distance = float('inf')
        closest_keypoint = None
        
        # Calculate distance to each keypoint
        for kp in keypoints:
            kp_x, kp_y = kp.pt
            distance = np.sqrt((x - kp_x)**2 + (y - kp_y)**2)
            if distance < closest_distance:
                closest_distance = distance
                closest_keypoint = kp
        tracked_keypoint = closest_keypoint

# Function to capture video from webcam and show keypoints
def main():
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    cv2.namedWindow('Webcam with Keypoints')
    cv2.setMouseCallback('Webcam with Keypoints', mouse_click)
    
    global tracked_keypoint, prev_gray
    
    recording = False
    writer = None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_filename = 'output.mp4'
    
    prev_time = time.time()
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Detect keypoints and draw them on the frame
        frame_with_keypoints, gray, feature_extraction_time = detect_keypoints(frame)
        
        # If tracked keypoint is found, draw circle around it
        if tracked_keypoint is not None:
            x, y = tracked_keypoint.pt
            cv2.circle(frame_with_keypoints, (int(x), int(y)), 10, (0, 255, 0), 2) # Green color
            # Perform optical flow to track the keypoint
            new_tracked_keypoint, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, np.array([tracked_keypoint.pt], dtype=np.float32), None)
            if status[0] == 1:
                tracked_keypoint = cv2.KeyPoint(new_tracked_keypoint[0][0], new_tracked_keypoint[0][1], 10)
        
        # Store the current frame's grayscale image for the next iteration
        prev_gray = gray.copy()
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        kp_count = len(keypoints)
        
        # Overlay telemetry information
        cv2.putText(frame_with_keypoints, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_with_keypoints, f"Feature Extraction Time: {feature_extraction_time:.2f} ms", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_with_keypoints, f"Resolution: {frame_width}x{frame_height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_with_keypoints, f"Detected Keypoints: {kp_count}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Display the resulting frame
        cv2.imshow('Webcam with Keypoints', frame_with_keypoints)
        
        # Record video if spacebar is pressed
        key = cv2.waitKey(1)
        if key == ord(' '):
            if not recording:
                # Start recording
                writer = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))
                recording = True
                print("Recording started.")
            else:
                # Stop recording
                if writer is not None:
                    writer.release()
                    recording = False
                    print("Recording stopped. Video saved as", out_filename)
        
        # Write frame to video file if recording
        if recording:
            writer.write(frame_with_keypoints)
        
        # Check for 'q' key press to exit
        if key & 0xFF == ord('q'):
            break
    
    # Release the capture
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
