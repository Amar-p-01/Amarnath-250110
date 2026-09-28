from ultralytics import YOLO

def main():
    # Load backbone and neck features from your previous VisDrone checkpoint
    # Ultralytics will automatically adapt the detection head to SARD's class count
    model = YOLO('/home/aditi-kesari/CV_bootcamp/UNET+YOLO/YOLO/runs/detect/yolo_pretrained_project/pretrained_run/weights/best.pt') 

    print("Starting fine-tuning on SARD at 640px resolution...")
    
    results = model.train(
        data='SARD/search-and-rescue/data.yaml', 
        epochs=60,
        imgsz=640,              # Higher resolution preserves pixel clusters of distant humans
        batch=4,                # Reduced from 8 to fit within 4GB VRAM at 640px
        device=0,
        
        # Aerial-specific augmentations
        degrees=15.0,           # Random rotation up to +/- 15 degrees
        flipud=0.5,             # 50% chance of vertical flipping
        fliplr=0.5,             # 50% chance of horizontal flipping
        scale=0.7,              # Altitude variation
        close_mosaic=10,        # Turn off mosaic for final epochs to sharpen box boundaries
        
        project='yolo_sard_project', 
        name='sard_640_augmented'
    )

if __name__ == "__main__":
    main()
