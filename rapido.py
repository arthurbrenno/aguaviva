import os
import cv2
import questionary
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("weights/nano.pt")

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7
PADDING = 5  # Padding value in pixels

# Ask the user for the mode
mode = questionary.select(
    "What would you like to process?",
    choices=["Image", "Video"]
).ask()

if mode == "Image":
    # List available images in the 'images' folder
    image_folder = "images"
    images = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not images:
        print("No images found in the 'images' folder.")
    else:
        # Let the user select an image
        selected_image = questionary.select(
            "Select an image to process:",
            choices=images
        ).ask()

        # Read and process the selected image
        image_path = os.path.join(image_folder, selected_image)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Error: Could not load image at {image_path}")
        else:
            # Perform inference
            results = model.predict(frame)
            
            # Draw bounding boxes and label detections with confidence > 70%
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                confidence = result.conf.item()
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Apply padding and ensure it stays within the image bounds
                    height, width, _ = frame.shape
                    x1 = max(0, x1 - PADDING)
                    y1 = max(0, y1 - PADDING)
                    x2 = min(width - 1, x2 + PADDING)
                    y2 = min(height - 1, y2 + PADDING)

                    # Adjust label position dynamically
                    label = f"marine debris ({confidence:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    label_x = x1
                    label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10

                    # Draw the bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (label_x, label_y - label_size[1] - 5), 
                                  (label_x + label_size[0] + 5, label_y + 5), (0, 255, 0), -1)
                    cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Display the image
            cv2.imshow("Image Detection", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

elif mode == "Video":
    # Ask for the video path
    video_path = input("Enter the path to the video: ").strip()
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not load video at {video_path}")
    else:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform inference
            results = model.predict(frame)

            # Draw bounding boxes and label detections with confidence > 70%
            for result in results[0].boxes:
                x1, y1, x2, y2 = map(int, result.xyxy[0])
                confidence = result.conf.item()
                
                if confidence >= CONFIDENCE_THRESHOLD:
                    # Apply padding and ensure it stays within the frame bounds
                    height, width, _ = frame.shape
                    x1 = max(0, x1 - PADDING)
                    y1 = max(0, y1 - PADDING)
                    x2 = min(width - 1, x2 + PADDING)
                    y2 = min(height - 1, y2 + PADDING)

                    # Adjust label position dynamically
                    label = f"marine debris ({confidence:.2f})"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    label_x = x1
                    label_y = y1 - 10 if y1 - 10 > label_size[1] else y1 + label_size[1] + 10

                    # Draw the bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(frame, (label_x, label_y - label_size[1] - 5), 
                                  (label_x + label_size[0] + 5, label_y + 5), (0, 255, 0), -1)
                    cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Display the frame in a window
            cv2.imshow("Video Detection", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

else:
    print("Invalid mode selected.")
