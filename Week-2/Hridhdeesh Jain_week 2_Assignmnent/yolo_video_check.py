import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
source = 'finalvideo.mp4' 
cap = cv2.VideoCapture(source)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps: 
    fps = 30.0

# Define video writer codec and output file
output_path = "survivor_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Processing and saving video to '{output_path}'")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run direct inference filtering ONLY for 'person' (class 0)
    results = model.predict(frame, classes=[0], conf=0.5, verbose=False)

    # Render bounding boxes on frame
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    # Display live feed
    cv2.imshow("Direct Survivor Detection", annotated_frame)

    # Press 'q' to stop recording and exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video saved successfully as {output_path}")