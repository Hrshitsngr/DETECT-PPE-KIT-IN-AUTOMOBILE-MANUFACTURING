import cv2
from ultralytics import YOLO
import os
import time

# Load YOLOv8 trained model
model_path = "C:/Users/dell/Downloads/firstmilestone/best.pt"
model = YOLO(model_path)

# Create folder to save violations
violation_folder = "violations"
os.makedirs(violation_folder, exist_ok=True)

# Get video files from the folder
video_folder = "C:/Users/dell/Downloads/firstmilestone/videoo"
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

# Process videos
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        # Extract bounding boxes for people, helmets, and safety vests
        people_boxes = [box.xyxy[0].tolist() for box in results[0].boxes if model.names[int(box.cls)] == "Person"]
        helmet_boxes = [box.xyxy[0].tolist() for box in results[0].boxes if model.names[int(box.cls)] == "Hardhat"]
        vest_boxes = [box.xyxy[0].tolist() for box in results[0].boxes if model.names[int(box.cls)] == "Safety Vest"]

        # Check for people without helmets
        for person_box in people_boxes:
            px1, py1, px2, py2 = map(int, person_box)
            helmet_found = any(
                hx1 > px1 and hy1 > py1 and hx2 < px2 and hy2 < py2
                for hx1, hy1, hx2, hy2 in [map(int, helmet) for helmet in helmet_boxes]
            )

            vest_found = any(
                vx1 > px1 and vy1 > py1 and vx2 < px2 and vy2 < py2
                for vx1, vy1, vx2, vy2 in [map(int, vest) for vest in vest_boxes]
            )

            if not helmet_found:
                cv2.putText(frame, "No Helmet!", (px1, py1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 3)

                # Save violation frame
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(os.path.join(violation_folder, f"no_helmet_{timestamp}.jpg"), frame)

            if not vest_found:
                cv2.putText(frame, "No Vest!", (px1, py2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 165, 0), 3)

        # Display FPS
        fps = int(1 / (time.time() - prev_time))
        prev_time = time.time()
        cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("PPE Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

# Process each video
for video_file in video_files:
    process_video(os.path.join(video_folder, video_file))

cv2.destroyAllWindows()
