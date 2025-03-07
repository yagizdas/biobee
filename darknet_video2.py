#!/usr/bin/env python3

from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import signal
import sys

from picamera2 import Picamera2

# Initialize Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# Global variables
is_running = True
capture_thread = None
inference_thread = None
drawing_thread = None
frame_delay = 30  # Delay in milliseconds (default is 30 for normal speed)

# Get video dimensions from Picamera2 configuration
video_width = config["main"]["size"][0]
video_height = config["main"]["size"][1]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="window inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default="yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    parser.add_argument("--gpu_index", type=int, default=0,
                        help="GPU index to use for processing")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))


def set_saved_video(output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = 30  # Using a fixed FPS since we don't have cap.get(cv2.CAP_PROP_FPS)
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def convert2relative(bbox):
    x, y, w, h = bbox
    _height = darknet_height
    _width = darknet_width
    return x / _width, y / _height, w / _width, h / _height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)
    image_h, image_w, __ = image.shape
    orig_x = int(x * image_w)
    orig_y = int(y * image_h)
    orig_width = int(w * image_w)
    orig_height = int(h * image_h)
    bbox_converted = (orig_x, orig_y, orig_width, orig_height)
    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)
    image_h, image_w, __ = image.shape
    orig_left = int((x - w / 2.) * image_w)
    orig_right = int((x + w / 2.) * image_w)
    orig_top = int((y - h / 2.) * image_h)
    orig_bottom = int((y + h / 2.) * image_h)
    if orig_left < 0: orig_left = 0
    if orig_right > image_w - 1: orig_right = image_w - 1
    if orig_top < 0: orig_top = 0
    if orig_bottom > image_h - 1: orig_bottom = image_h - 1
    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)
    return bbox_cropping


def signal_handler(sig, frame):
    global is_running, capture_thread, inference_thread, drawing_thread
    is_running = False
    
    # Join threads if they exist
    if capture_thread is not None and capture_thread.is_alive():
        capture_thread.join()
    if inference_thread is not None and inference_thread.is_alive():
        inference_thread.join()
    if drawing_thread is not None and drawing_thread.is_alive():
        drawing_thread.join()
    
    # Stop the camera
    picam2.stop()
    
    # Close windows
    cv2.destroyAllWindows()
    
    # Exit the program
    sys.exit(0)


def video_capture(frame_queue, darknet_image_queue):
    global is_running, frame_delay
    
    while is_running:
        # Capture frame from picamera2
        frame = picam2.capture_array()
        
        # Handle key presses for speed control
        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            is_running = False
        elif key == 82:  # Arrow Up Key
            frame_delay = max(1, frame_delay - 5)  # Decrease delay, speed up
        elif key == 84:  # Arrow Down Key
            frame_delay += 5  # Increase delay, slow down
        
        # Convert the frame from RGB (picamera2 format) to BGR (OpenCV format)
        # Note: picamera2 with RGB888 format already gives us RGB format
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Resize the frame to the dimensions expected by Darknet
        frame_resized = cv2.resize(frame, (darknet_width, darknet_height), interpolation=cv2.INTER_LINEAR)
        
        # Put the original BGR frame in the frame queue for drawing
        frame_queue.put(frame_bgr)
        
        # Create a Darknet image from the resized frame
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        
        # Put the Darknet image in the Darknet image queue
        darknet_image_queue.put(img_for_detect)


def inference(darknet_image_queue, detections_queue, fps_queue):
    global is_running
    
    while is_running:
        if darknet_image_queue.empty():
            continue
            
        darknet_image = darknet_image_queue.get()  # Retrieve an image from the queue
        
        prev_time = time.time()  # Record the time before detection
        
        # Perform object detection on the image
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        
        detections_queue.put(detections)  # Put the detections in the detections queue
        
        # Calculate FPS
        fps = int(1 / (time.time() - prev_time))
        fps_queue.put(fps)  # Put the FPS value in the fps queue
        print("FPS: {}".format(fps))  # Print the FPS
        
        # Print detections if extended output is enabled
        darknet.print_detections(detections, args.ext_output)
        
        darknet.free_image(darknet_image)  # Free the memory of the Darknet image


def drawing(frame_queue, detections_queue, fps_queue):
    global is_running
    random.seed(3)  # Ensure consistent colors for bounding boxes across runs
    
    # Initialize video writer if an output filename is specified
    video = None
    if args.out_filename:
        video = set_saved_video(args.out_filename, (video_width, video_height))
    
    while is_running:
        if frame_queue.empty() or detections_queue.empty() or fps_queue.empty():
            continue
            
        frame = frame_queue.get()  # Retrieve a frame from the queue
        detections = detections_queue.get()  # Retrieve detections for the frame
        fps = fps_queue.get()  # Retrieve the FPS value
        
        detections_adjusted = []
        
        if frame is not None:
            # Adjust each detection to the original frame size and add to list
            for label, confidence, bbox in detections:
                bbox_adjusted = convert2original(frame, bbox)
                detections_adjusted.append((str(label), confidence, bbox_adjusted))
            
            # Draw bounding boxes on the frame
            image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
            
            # Add FPS text to the image
            cv2.putText(image, f"FPS: {fps}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            if not args.dont_show:
                cv2.imshow('Inference', image)  # Display the frame
            
            # Check if the 'q' key is pressed to stop the process
            if cv2.waitKey(1) & 0xFF == ord('q'):
                is_running = False
                break
            
            # Write the frame to the output video file if specified
            if video is not None:
                video.write(image)
    
    # Release resources
    if video is not None:
        video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Register the signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Parse command-line arguments
    args = parse_args()
    
    # Set GPU and perform argument checks
    darknet.set_gpu(args.gpu_index)
    check_arguments_errors(args)
    
    # Load YOLO network
    network, class_names, class_colors = darknet.load_network(
        args.config_file, args.data_file, args.weights, batch_size=1
    )
    
    # Get network dimensions
    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    
    # Create queues for thread communication
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    
    # Start threads
    capture_thread = Thread(target=video_capture, args=(frame_queue, darknet_image_queue))
    inference_thread = Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue))
    drawing_thread = Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue))
    
    capture_thread.start()
    inference_thread.start()
    drawing_thread.start()
    
    try:
        # Keep the main thread alive
        while is_running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        # Handle Ctrl+C
        is_running = False
    
    # Wait for threads to finish
    if capture_thread.is_alive():
        capture_thread.join()
    if inference_thread.is_alive():
        inference_thread.join()
    if drawing_thread.is_alive():
        drawing_thread.join()
    
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
    
    # Flush and close standard outputs
    sys.stdout.flush()
    sys.stderr.flush()