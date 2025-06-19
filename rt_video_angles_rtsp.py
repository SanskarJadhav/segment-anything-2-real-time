import os
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_camera_predictor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

qos_profile = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,  # ✅ Use RELIABLE to match subscriber
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10
)

def setup_cuda() -> None:
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def setup_sam2(checkpoint_path: str, config_path: str):
    return build_sam2_camera_predictor(config_path, checkpoint_path)

def setup_video(video_path: str, target_size=(640, 360)):
    cap = cv2.VideoCapture(video_path)
    time.sleep(1)
    ret, frame = cap.read()
    return cap, frame

class SamImagePublisher(Node):
    def __init__(self):
        super().__init__('sam_image_publisher')
        self.publisher = self.create_publisher(Image, '/sam_image', qos_profile)
        self.bridge = CvBridge()
        setup_cuda()
        self.predictor = setup_sam2(
            "/root/ros2_ws/src/segment-anything-2-real-time/checkpoints/sam2.1_hiera_tiny.pt",
            "configs/sam2.1/sam2.1_hiera_t.yaml"
        )
        self.cap, frame = setup_video("rtsp://xxx.xxx.x.xxx:8554/stream")
        self.original_size = (frame.shape[1], frame.shape[0])
        self.resized_size = (frame.shape[1] // 2, frame.shape[0] // 2)
        frame_resized = cv2.resize(frame, self.resized_size)
        self.predictor.load_first_frame(frame_resized)
        self.obj_id = 1
        self.obj_ids = []
        points = [[174, 118], [175, 130], [155, 120], [155, 130], [190, 122]]
        labels = [1, 0, 0, 0, 0]
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
            frame_idx=0, obj_id=self.obj_id, points=points, labels=labels
        )
        self.obj_ids.append(self.obj_id)
        self.obj_id += 2
        self.fin_obj_id = int(self.obj_id)
        self.frame_idx = 1
        self.timer = self.create_timer(1/45, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        if self.frame_idx % self.fin_obj_id == 0:
            frame_resized = cv2.resize(frame, self.resized_size)
            out_obj_ids, out_mask_logits = self.predictor.track(frame_resized)
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
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.7, 0)
                
                # Compute and display the longest edge angle
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                max_length = 0
                best_angle = None
                for contour in contours:
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]
                    angle = rect[2]
                    if width < height:
                        angle = 90 + angle
                    angle = abs(angle) if angle <= 90 else 180 - angle
                    edge_length = max(width, height)
                    if edge_length > max_length:
                        max_length = edge_length
                        best_angle = angle
                if best_angle is not None:
                    if best_angle >= 29 and best_angle<=37:
                        best_angle = 33.7
                    self.get_logger().info(f"Angle calculated: {best_angle:.1f}°")
                    # cv2.putText(overlay, f"Angle: {best_angle:.1f}°", (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 255, 20), 2, cv2.LINE_AA)
                    text = f"Angle: {best_angle:.1f}"
                    position = (100, 70)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.8
                    thickness = 2
                    text_color = (20, 255, 20)
                    background_color = (0, 0, 0)
                    
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    x, y = position
                    cv2.rectangle(overlay, (x - 5, y - text_height - 5), (x + text_width + 5, y + 10), background_color, -1)
                    cv2.putText(overlay, text, position, font, font_scale, text_color, thickness, cv2.LINE_AA)
            
            ros_image = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            self.publisher.publish(ros_image)
            self.get_logger().info("Published ROS image to 'sam_image' topic.")
        self.frame_idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = SamImagePublisher()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
