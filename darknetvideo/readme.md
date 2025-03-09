# Darknet/YOLO Command Guide for Python

## Video/Webcam/Display Commands

### Process a Video File
    python darknet_video.py --input 2.mp4 --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data

### Use Display for Real-Time Detection
    python darknet_video.py --input 0 --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data

### Use Webcam for Real-Time Detection
    python darknet_video.py --input 1 --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data

### Save Processed Video
    python darknet_video.py --input path/to/your/video.mp4 --out_filename processed_video.avi --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data

### Use Different YOLO Model
    python darknet_video.py --input path/to/your/video.mp4 --weights path/to/your/weights.weights --config_file path/to/your/config.cfg --data_file path/to/your/data.data

## Parameters

### Select a Different GPU
    --gpu_index 1

### Change Detection Threshold
    --thresh 0.3

### Run in Headless Mode
    --dont_show

### Display Extended Output
    --ext_output

### Controls
- Right-click in console to stop video
- Left-click in console to resume
- Press 'q' key to stop video

## Image Detection Commands

### Single Image Detection
    python darknet_images.py --input dog.jpg --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data --gpu

### Batch Image Detection from Text File
    python darknet_images.py --input train.txt --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data --gpu

### Batch Image Detection from Folder
    python darknet_images.py --input /path/to/folder --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data --gpu

### Change Detection Threshold for Images
    python darknet_images.py --input dog.jpg --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data --gpu --thresh 0.3

### Apply A Different Non-Maximum Suppression (NMS)
    python darknet_images.py --input dog.jpg --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data --gpu --nms_thresh 0.45

### Save Detection Labels
    python darknet_images.py --input train.txt --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data --gpu --save_labels

### Display Extended Output for Images
    python darknet_images.py --input dog.jpg --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data --gpu --ext_output

### Run Image Detection in Headless Mode
    python darknet_images.py --input dog.jpg --weights yolov4.weights --config_file yolov4.cfg --data_file obj.data --gpu --dont_show

## Error Handling and Notes
- Ensure correct paths, file names, and formats.
- Use Ctrl+C to stop execution in console.
- Press `q` to quit display windows.

Note:
> Command usability depends on `darknet_images.py` implementation and Darknet version.

