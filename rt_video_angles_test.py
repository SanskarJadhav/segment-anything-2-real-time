import os
import time
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_camera_predictor

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


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
    time.sleep(2)
    ret, frame = cap.read()
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


# ROS2 Publisher Node
class FramePublisher(Node):
    def __init__(self):
        super().__init__('frame_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/color/image_angle', 10)
        self.bridge = CvBridge()

    def publish_frame(self, frame):
        """Publish the overlaid frame as a ROS2 topic."""
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.publisher_.publish(msg)


def main():
    """Main execution function for real-time video segmentation with multiple objects."""
    # Initialize ROS2
    rclpy.init()
    publisher_node = FramePublisher()

    # Initialize CUDA and SAM2
    setup_cuda()
    predictor = setup_sam2(
        "checkpoints/sam2.1_hiera_small.pt",
        "configs/sam2.1/sam2.1_hiera_s.yaml"
    )

    # Setup video capture
    cap, frame = setup_video("rtsp://192.168.0.200:8554/stream")
    original_size = (frame.shape[1], frame.shape[0])
    resized_size = (frame.shape[1] // 2, frame.shape[0] // 2)
    frame_resized = cv2.resize(frame, resized_size)

    # Define the output video filename and codec
    output_filename = "segmented_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_filename, fourcc, fps, resized_size)

    # Initialize first frame and get user prompts for multiple objects
    predictor.load_first_frame(frame_resized)
    using_point = get_prompt_type()

    obj_id = 1
    obj_ids = []
    while True:
        add_more = input("Add an object? (y/n): ").strip().lower()
        if add_more != 'y':
            break

        if using_point:
            point_selector = PointSelector()
            points, labels = point_selector.get_points(frame_resized)
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=0, obj_id=obj_id, points=points, labels=labels
            )
        else:
            bbox_drawer = BoundingBoxDrawer()
            bbox = bbox_drawer.get_bbox(frame_resized)
            _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                frame_idx=0, obj_id=obj_id, bbox=bbox
            )

        obj_ids.append(obj_id)
        obj_id += 1

    # Main video processing loop
    frame_idx = 1
    fin_obj_id = int(obj_id)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, resized_size)
            out_obj_ids, out_mask_logits = predictor.track(frame_resized)

            colors = [
                (255, 0, 0, 100), (0, 255, 0, 100), (0, 0, 255, 100), (255, 255, 0, 100),
                (255, 0, 255, 100), (0, 255, 255, 100), (128, 0, 128, 100), (0, 128, 128, 100)
            ]

            overlay = frame_resized.copy()
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                h, w = mask.shape[-2:]
                mask = mask.reshape(h, w).astype(np.uint8)
                color = colors[i % len(colors)]

                colored_mask = np.zeros_like(frame_resized, dtype=np.uint8)
                for j in range(3):
                    colored_mask[:, :, j] = np.where(mask == 1, color[j], 0)

                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.6, 0)

            # Display the overlaid frame
            cv2.imshow('Segmented', overlay)

            # Write the frame to the output video file
            out.write(overlay)

            # Publish the frame to ROS2 topic
            publisher_node.publish_frame(overlay)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        out.release()
        print(f"Video saved as: {output_filename}")
        publisher_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

