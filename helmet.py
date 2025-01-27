import cv2
import numpy as np
import torch
from torchvision import transforms
import os

# Load YOLO for person detection
person_weights = "C:/Users/dell/Downloads/firstmilestone/yolov3.weights"
person_config = "C:/Users/dell/Downloads/firstmilestone/yolov3.cfg"
person_net = cv2.dnn.readNet(person_weights, person_config)

# Use the COCO dataset class names for person detection
coco_names_path = "C:/Users/dell/Downloads/firstmilestone/coco.names"
with open(coco_names_path, "r") as file:
    person_classes = [line.strip() for line in file.readlines()]

# Extract output layers
person_layer_names = person_net.getLayerNames()
person_output_layers = [person_layer_names[i - 1] for i in person_net.getUnconnectedOutLayers()]

# Load the PyTorch helmet detection model
helmet_model_path = "C:/Users/dell/Downloads/firstmilestone/yolo11_helmet_detection_ai.pt"
helmet_model = torch.load(helmet_model_path)
helmet_model.eval()

# Define a transform for the input image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load all video files from the 'videos' directory
video_folder = "C:/Users/dell/Downloads/firstmilestone/videoo"
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        # Person detection
        blob_person = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        person_net.setInput(blob_person)
        person_outs = person_net.forward(person_output_layers)

        person_boxes = []
        for out in person_outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and class_id == person_classes.index("person"):
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    person_boxes.append((x, y, w, h))

        # Helmet detection
        helmet_boxes = []
        for (x, y, w, h) in person_boxes:
            person_roi = frame[y:y + h, x:x + w]
            if person_roi.size == 0:
                continue

            input_image = transform(person_roi).unsqueeze(0)
            with torch.no_grad():
                outputs = helmet_model(input_image)

            # Parse the output for helmet detection (assuming the model provides bounding boxes and scores)
            for box in outputs["boxes"]:
                score = outputs["scores"][outputs["boxes"].tolist().index(box)]
                if score > 0.5:
                    hx, hy, hw, hh = [int(coord) for coord in box.tolist()]
                    helmet_boxes.append((hx + x, hy + y, hw, hh))

        # Check if each person has a helmet
        for (x, y, w, h) in person_boxes:
            person_has_helmet = False
            for (hx, hy, hw, hh) in helmet_boxes:
                # Check if helmet is within the person's bounding box
                if hx > x and hy > y and (hx + hw) < (x + w) and (hy + hh) < (y + h):
                    person_has_helmet = True
                    break

            # Draw bounding boxes
            if person_has_helmet:
                color = (0, 255, 0)  # Green for wearing helmet
            else:
                color = (0, 0, 255)  # Red for not wearing helmet

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Display the frame
        cv2.imshow("Helmet Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

cv2.destroyAllWindows()
