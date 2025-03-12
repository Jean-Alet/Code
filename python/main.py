# Libraries you need to install 
# pip install ultralytics opencv-python numpy torch torchvision torchaudio
# pip install onnxruntime deep_sort_realtime


from ultralytics import YOLO
import cv2

# Load YOLOv8 pre-trained model
model = YOLO("yolov8n.pt")  # Use 'yolov8m.pt' or 'yolov8l.pt' for better accuracy

# Open video file
video_path = "D:/Code/python/vb1.avi"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = model(frame)

    # Draw detections
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index

            if cls == 0:  # YOLO class 0 = person
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Player {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Volleyball Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()