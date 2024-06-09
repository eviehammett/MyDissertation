import cv2
import random
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('ASLData.pt')

# Function to process frames and detect hand signs using YOLOv8
def detect_sign_language(frame, target_letter):
    frame = cv2.flip(frame, 1)
    
    # Perform YOLOv8 detection
    results = model(frame)
    
    detected_label = None
    
    # Draw bounding boxes and labels on the frame
    for result in results:
        for bbox in result.boxes:
            x_min, y_min, x_max, y_max = map(int, bbox.xyxy[0])
            label = result.names[int(bbox.cls[0])]
            confidence = bbox.conf[0]
            
            # Draw bounding box
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            # Put label and confidence score
            cv2.putText(frame, f"{label} {confidence:.2f}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 0, 0), 2)
            
            detected_label = label
    
    # Display the target letter on the frame
    cv2.putText(frame, f"Sign this letter: {target_letter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0), 2, cv2.LINE_AA)
    
    if detected_label is not None:
        if detected_label == target_letter:
            feedback = "Correct! Press F to change letters."
        else:
            feedback = "Try again or press F to change letters!"
        cv2.putText(frame, feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                    (255, 0, 255), 2, cv2.LINE_AA)
    
    return frame

# Initialize video capture
camera = cv2.VideoCapture(0)

# Generate a random letter from A-Y
random_letter = chr(random.randint(ord('A'), ord('Y')))

# Display video in a window
while True:
    success, frame = camera.read()
    if not success:
        break
    
    frame = detect_sign_language(frame, random_letter)
    cv2.imshow('Sign Language Detection', frame)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    
    # Exit on 'q' key press
    if key == ord('q'):
        break
    # Generate a new random letter on 'F' key press
    elif key == ord('f'):
        random_letter = chr(random.randint(ord('A'), ord('Y')))
    
# Release resources
camera.release()
cv2.destroyAllWindows()
