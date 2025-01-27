import cv2
from ultralytics import YOLO
import os
import time

# Load YOLOv8 trained model
model_path = "C:/Users/dell/Downloads/firstmilestone/best.pt"
model = YOLO(model_path)

# Folder to save violations
violation_folder = "violations"  
os.makedirs(violation_folder, exist_ok=True)

# Video folder path
video_folder = "C:/Users/dell/Downloads/firstmilestone/videoo"
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

# Process all videos
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)

        people_boxes = []
        helmet_boxes = []

        for result in results:
            boxes = result.boxes.xyxy  # Bounding box coordinates
            confidences = result.boxes.conf  # Confidence scores
            class_ids = result.boxes.cls  # Class indices

            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(int, box)  # Bounding box
                label = model.names[int(class_id)]  # Class name

                if label == "Person":
                    people_boxes.append([x1, y1, x2, y2])
                elif label == "Hardhat":
                    helmet_boxes.append([x1, y1, x2, y2])

                # Draw bounding boxes
                color = (0, 255, 0) if label == "Hardhat" else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check for people without helmets
        for px1, py1, px2, py2 in people_boxes:
            helmet_found = any(hx1 > px1 and hy1 > py1 and hx2 < px2 and hy2 < py2 for hx1, hy1, hx2, hy2 in helmet_boxes)
            if not helmet_found:
                cv2.putText(frame, "No Helmet!", (px1, py1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)

                # Save violation frames
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(os.path.join(violation_folder, f"no_helmet_{timestamp}.jpg"), frame)

        # FPS counter
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("PPE Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()
