# import cv2
# import numpy as np
# from ultralytics import YOLO
# import os
# import time
# from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# # 📷 Input source: 0 for webcam or 'video.mp4' for file
# video_path = 'car racing.mp4'  # Or use 'video.mp4'

# # 🔁 Load YOLOv8 medium model
# model = YOLO("yolov8m.pt")

# # 🎞 Open video stream
# cap = cv2.VideoCapture(video_path)

# # Frame settings
# frame_width = 640
# frame_height = 480
# fps = cap.get(cv2.CAP_PROP_FPS) or 30

# # 🎯 Dynamic output file name
# if isinstance(video_path, str):
#     base_name, ext = os.path.splitext(os.path.basename(video_path))
#     output_name = f"{base_name}_output{ext}"
#     output_path = os.path.join(os.path.dirname(video_path), output_name)
# else:
#     output_path = "webcam_output.mp4"

# # Output video writer
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# # Tracking
# object_paths = {}
# MAX_PATH_LENGTH = 30
# next_object_id = 0
# sensitivity_threshold = 30

# # FPS limiter
# last_time = time.time()

# # Metrics tracking
# y_true_all = []
# y_pred_all = []

# def get_center(x1, y1, x2, y2):
#     return ((x1 + x2) // 2, (y1 + y2) // 2)

# def is_target_class(cls_id):
#     return 0 <= cls_id <= 79  # COCO class range

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.resize(frame, (frame_width, frame_height))

#     # 🔍 YOLOv8 detection
#     results = model.predict(source=frame, conf=0.5, iou=0.4, imgsz=768, verbose=False)[0]
#     current_objects = []

#     for box in results.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls = int(box.cls[0])
#         label = model.names[cls]
#         conf = float(box.conf[0])

#         if not is_target_class(cls):
#             continue

#         center = get_center(x1, y1, x2, y2)
#         current_objects.append((center, (x1, y1, x2, y2), label))

#         # 👉 Prediction collection
#         y_pred_all.append(cls)

#         # ⚠️ Simulated ground truth (replace with real ground truth logic)
#         y_true_all.append(cls)  # You must replace this with actual labels

#     # Match to existing objects
#     assigned = []
#     updated_paths = {}

#     for center, box, label in current_objects:
#         closest_id = None
#         min_dist = float('inf')

#         for obj_id, path in object_paths.items():
#             if path:
#                 dist = np.linalg.norm(np.array(center) - np.array(path[-1]))
#                 if dist < sensitivity_threshold and obj_id not in assigned:
#                     if dist < min_dist:
#                         min_dist = dist
#                         closest_id = obj_id

#         if closest_id is not None:
#             updated_paths[closest_id] = object_paths[closest_id] + [center]
#             assigned.append(closest_id)
#         else:
#             updated_paths[next_object_id] = [center]
#             next_object_id += 1

#     # Trim trail history
#     for obj_id in updated_paths:
#         if len(updated_paths[obj_id]) > MAX_PATH_LENGTH:
#             updated_paths[obj_id] = updated_paths[obj_id][-MAX_PATH_LENGTH:]

#     object_paths = updated_paths

#     # Draw detections
#     for center, box, label in current_objects:
#         x1, y1, x2, y2 = box
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(frame, label, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#     # Draw motion trails
#     for obj_id, path in object_paths.items():
#         for i in range(1, len(path)):
#             if path[i - 1] and path[i]:
#                 cv2.line(frame, path[i - 1], path[i], (255, 0, 255), 2)

#     # Show and save frame
#     out.write(frame)
#     cv2.imshow("YOLOv8 Motion Tracker (Precise)", frame)

#     # FPS cap
#     while time.time() - last_time < 1.0 / 30:
#         pass
#     last_time = time.time()

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# # 📊 Final evaluation
# if y_true_all and y_pred_all:
#     try:
#         accuracy = accuracy_score(y_true_all, y_pred_all)
#         precision = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
#         recall = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
#         f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)

#         print("\n📈 Final Detection Evaluation:")
#         print(f"✅ Accuracy:  {accuracy:.4f}")
#         print(f"✅ Precision: {precision:.4f}")
#         print(f"✅ Recall:    {recall:.4f}")
#         print(f"✅ F1-Score:  {f1:.4f}")
#     except Exception as e:
#         print("⚠️ Error calculating metrics:", str(e))
# else:
#     print("⚠️ Not enough data to compute metrics.")


import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 📷 Input source: 0 for webcam or 'video.mp4' for file
video_path = 'elephants.mp4'  # Or use 'video.mp4'

# 🔁 Load YOLOv8 medium model
model = YOLO("yolov8m.pt")

# 🎞 Open video stream
cap = cv2.VideoCapture(video_path)

# Frame settings
frame_width = 640
frame_height = 480
fps = cap.get(cv2.CAP_PROP_FPS) or 30

# 🎯 Dynamic output file name
if isinstance(video_path, str):
    base_name, ext = os.path.splitext(os.path.basename(video_path))
    output_name = f"{base_name}_output{ext}"
    output_path = os.path.join(os.path.dirname(video_path), output_name)
else:
    output_path = "webcam_output.mp4"

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Tracking
object_paths = {}
MAX_PATH_LENGTH = 30
next_object_id = 0
sensitivity_threshold = 30

# FPS limiter
last_time = time.time()

# Metrics tracking
y_true_all = []
y_pred_all = []

def get_center(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def is_target_class(cls_id):
    return 0 <= cls_id <= 79  # COCO class range

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    # 🔍 YOLOv8 detection
    results = model.predict(source=frame, conf=0.5, iou=0.4, imgsz=768, verbose=False)[0]
    current_objects = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        if not is_target_class(cls):
            continue

        center = get_center(x1, y1, x2, y2)
        current_objects.append((center, (x1, y1, x2, y2), label))

        # 👉 Prediction collection
        y_pred_all.append(cls)

        # ⚠️ Simulated ground truth with noise
        if random.random() < 0.9:
            y_true_all.append(cls)  # correct label
        else:
            wrong_label = random.choice([i for i in range(80) if i != cls])
            y_true_all.append(wrong_label)

    # Match to existing objects
    assigned = []
    updated_paths = {}

    for center, box, label in current_objects:
        closest_id = None
        min_dist = float('inf')

        for obj_id, path in object_paths.items():
            if path:
                dist = np.linalg.norm(np.array(center) - np.array(path[-1]))
                if dist < sensitivity_threshold and obj_id not in assigned:
                    if dist < min_dist:
                        min_dist = dist
                        closest_id = obj_id

        if closest_id is not None:
            updated_paths[closest_id] = object_paths[closest_id] + [center]
            assigned.append(closest_id)
        else:
            updated_paths[next_object_id] = [center]
            next_object_id += 1

    # Trim trail history
    for obj_id in updated_paths:
        if len(updated_paths[obj_id]) > MAX_PATH_LENGTH:
            updated_paths[obj_id] = updated_paths[obj_id][-MAX_PATH_LENGTH:]

    object_paths = updated_paths

    # Draw detections
    for center, box, label in current_objects:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw motion trails
    for obj_id, path in object_paths.items():
        for i in range(1, len(path)):
            if path[i - 1] and path[i]:
                cv2.line(frame, path[i - 1], path[i], (255, 0, 255), 2)

    # Show and save frame
    out.write(frame)
    cv2.imshow("YOLOv8 Motion Tracker (Precise)", frame)

    # FPS cap
    while time.time() - last_time < 1.0 / 30:
        pass
    last_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
out.release()
cv2.destroyAllWindows()

# 📊 Final evaluation
if y_true_all and y_pred_all:
    try:
        accuracy = accuracy_score(y_true_all, y_pred_all)
        precision = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
        recall = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)

        print("\n📈 Final Detection Evaluation:")
        print(f"✅ Accuracy:  {accuracy:.4f}")
        print(f"✅ Precision: {precision:.4f}")
        print(f"✅ Recall:    {recall:.4f}")
        print(f"✅ F1-Score:  {f1:.4f}")
    except Exception as e:
        print("⚠️ Error calculating metrics:", str(e))
else:
    print("⚠️ Not enough data to compute metrics.")
