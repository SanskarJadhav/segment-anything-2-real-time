import os
import time
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_camera_predictor


def setup_cuda() -> None:
    """Configure CUDA settings for optimal performance with SAM2."""
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def setup_sam2(checkpoint_path: str, config_path: str):
    return build_sam2_camera_predictor(config_path, checkpoint_path)


def setup_video(video_path: str, target_size: Tuple[int, int] = (640, 360)) -> Tuple:
    cap = cv2.VideoCapture(video_path)
    time.sleep(1)
    ret, frame = cap.read()
    # frame = cv2.resize(frame, target_size)
    return cap, frame


class BoundingBoxDrawer:
    """Handles drawing and capturing bounding box inputs from user."""
    
    def __init__(self):
        self.bbox = None
        self.drawing = False
        self.start_point = None
        self.temp_frame = None

    def draw_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            img_copy = self.temp_frame.copy()
            cv2.rectangle(img_copy, self.start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("First Frame - Draw BBox", img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.bbox = [self.start_point, (x, y)]
            cv2.rectangle(self.temp_frame, self.start_point, (x, y), (0, 255, 0), 2)
            cv2.imshow("First Frame - Draw BBox", self.temp_frame)

    def get_bbox(self, frame: np.ndarray) -> np.ndarray:
        self.temp_frame = frame.copy()
        window_name = "First Frame - Draw BBox"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.draw_callback)
        cv2.imshow(window_name, self.temp_frame)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        
        if self.bbox is None:
            raise ValueError("No bounding box was drawn.")
        return np.array([self.bbox], dtype=np.float32)


class PointSelector:
    """Handles selection of points for positive/negative prompts."""
    
    def __init__(self):
        self.points = []
        self.labels = []
        self.temp_frame = None

    def point_callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click for positive point
            self.points.append([x, y])
            self.labels.append(1)
            cv2.circle(self.temp_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("First Frame - Select Points", self.temp_frame)
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click for negative point
            self.points.append([x, y])
            self.labels.append(0)
            cv2.circle(self.temp_frame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("First Frame - Select Points", self.temp_frame)

    def get_points(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.temp_frame = frame.copy()
        window_name = "First Frame - Select Points"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.point_callback)
        print("Left click for positive points, right click for negative points. Press any key when done.")
        cv2.imshow(window_name, self.temp_frame)
        cv2.waitKey(0)
        cv2.destroyWindow(window_name)
        
        if not self.points:
            raise ValueError("No points were selected.")
        return np.array(self.points, dtype=np.float32), np.array(self.labels, dtype=np.int32)


def get_prompt_type() -> bool:
    while True:
        prompt_type = input("Select prompt type (1 for points, 2 for bbox): ").strip()
        if prompt_type in ['1', '2']:
            return prompt_type == '1'




def main():
    """Main execution function for real-time video segmentation with multiple objects."""
    
    # Initialize CUDA and SAM2
    setup_cuda()
    predictor = setup_sam2(
        "checkpoints/sam2.1_hiera_small.pt",
        "configs/sam2.1/sam2.1_hiera_s.yaml"
    )

    # Setup video capture
    # cap, frame = setup_video("notebooks/videos/20250218_170426.mp4")
    cap, frame = setup_video("rtsp://192.168.0.200:8554/stream")
    
    # Dynamically resize frame for prediction (50% of original size)
    original_size = (frame.shape[1], frame.shape[0])
    resized_size = (frame.shape[1] // 2, frame.shape[0] // 2)
    frame_resized = cv2.resize(frame, resized_size)

            # Define the output video filename and codec
    output_filename = "segmented_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get original FPS of the video
    out = cv2.VideoWriter(output_filename, fourcc, fps, resized_size)
        
        
    # Initialize first frame and get user prompts for multiple objects
    predictor.load_first_frame(frame_resized)
    
    obj_id = 1  # Initialize object ID counter
    obj_ids = []  # Store object IDs


    points = [[175, 120], [175, 130], [155, 120], [190, 120]]
    labels = [1, 0, 0, 0]
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
        frame_idx=0, obj_id=obj_id, points=points, labels=labels
    )
       
    obj_ids.append(obj_id)
    obj_id += 1
    
    # Main video processing loop
    frame_idx = 1
    fin_obj_id = int(obj_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if frame_idx % fin_obj_id == 1:
        
            # Dynamically resize frame for prediction (50% of original size)
            frame_resized = cv2.resize(frame, resized_size)
            
            # Track all objects
            out_obj_ids, out_mask_logits = predictor.track(frame_resized)
            
            # Define colors for different objects
            colors = [
                (255, 0, 0, 100), (0, 255, 0, 100), (0, 0, 255, 100), (255, 255, 0, 100),
                (255, 0, 255, 100), (0, 255, 255, 100), (128, 0, 128, 100), (0, 128, 128, 100)
            ]
            
            # Create an overlay for transparency
            overlay = frame_resized.copy()
            
            # Apply masks for each object with different transparent colors
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                h, w = mask.shape[-2:]
                mask = mask.reshape(h, w).astype(np.uint8)
                color = colors[i % len(colors)]  # Cycle through colors
                
                colored_mask = np.zeros_like(frame_resized, dtype=np.uint8)
                for j in range(3):
                    colored_mask[:, :, j] = np.where(mask == 1, color[j], 0)
                
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.6, 0)
                
                # Longest Edge and Angle Calculation
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Fit a minimum area rectangle around the contour
                    # Calculate the minimum area rectangle for the contour
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = box.astype(np.int32)

                    # Get the width and height of the box
                    width = rect[1][0]
                    height = rect[1][1]

                    # Calculate the angle and normalize it
                    angle = rect[2]
                    if width < height:
                        angle = 90 + angle  # Adjust for OpenCV's rectangle rotation

                    # Normalize to 0-90 degrees
                    angle = abs(angle)
                    if angle > 90:
                        angle = 180 - angle

                    # Display the longest edge and angle
                    cv2.putText(overlay, f"Angle: {angle:.1f}", 
                                (box[0][0], box[0][1] - 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.6, (20, 255, 20), 2, cv2.LINE_AA)

    
            
            cv2.imshow('Segmented', overlay)
            # Write the current frame to the output video
            out.write(overlay)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_idx += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    # Release the VideoWriter
    out.release()
    print(f"Video saved as: {output_filename}")


if __name__ == "__main__":
    main()
