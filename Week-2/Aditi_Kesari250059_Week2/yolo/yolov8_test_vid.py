import os
import glob
import cv2
from ultralytics import YOLO

def process_video_sequence(sequence_dir, model_path, output_path, conf=0.25, fps=30):
    # 1. Load the trained model
    print(f"Loading weights from: {model_path}")
    model = YOLO(model_path)

    # 2. Collect and sort frames chronologically
    frame_paths = sorted(
        glob.glob(os.path.join(sequence_dir, "*.jpg")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
    )

    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in: {sequence_dir}")

    # 3. Read first frame to determine video dimensions
    first_frame = cv2.imread(frame_paths[0])
    height, width, _ = first_frame.shape

    # 4. Initialize VideoWriter for MP4 output
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing {len(frame_paths)} frames from {sequence_dir} at {width}x{height}...")

    # 5. Stream each frame through YOLO and write to video
    for idx, frame_path in enumerate(frame_paths, start=1):
        frame = cv2.imread(frame_path)

        # Run inference on current frame
        results = model.predict(source=frame, conf=conf, verbose=False, device=0)

        # Render bounding boxes onto the frame
        annotated_frame = results[0].plot()

        # Write frame to video stream
        video_writer.write(annotated_frame)

        if idx % 50 == 0 or idx == len(frame_paths):
            print(f"Rendered [{idx}/{len(frame_paths)}] frames...")

    # Release resources
    video_writer.release()
    print(f"\nProcessing complete. Annotated video saved to: {output_path}")

def main():
    # Path to your fine-tuned SARD model weights
    model_path = '/home/aditi-kesari/CV_bootcamp/UNET+YOLO/YOLO/runs/detect/yolo_sard_project/sard_640_augmented/weights/best.pt'
    
    # Target sequence folder from VisDrone-VID
    sequence_dir = '/home/aditi-kesari/CV_bootcamp/UNET+YOLO/YOLO/VisDrone2019-VID-test-challenge/sequences/uav0000097_00000_v'
    
    # Destination for the compiled video
    output_path = 'runs/detect/annotated_uav0000097.mp4'

    process_video_sequence(
        sequence_dir=sequence_dir,
        model_path=model_path,
        output_path=output_path,
        conf=0.25,
        fps=30
    )

if __name__ == "__main__":
    main()
