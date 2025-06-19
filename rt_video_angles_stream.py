import cv2
import numpy as np
import torch
from flask import Flask, Response
from sam2.build_sam import build_sam2_camera_predictor

app = Flask(__name__)

def setup_cuda():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def setup_sam2(checkpoint_path: str, config_path: str):
    return build_sam2_camera_predictor(config_path, checkpoint_path)

def setup_video():
    return cv2.VideoCapture("rtsp://xxx.xxx.x.xxx:8554/stream")

predictor = None
cap = None

def generate_frames():
    global predictor, cap
    setup_cuda()
    predictor = setup_sam2("checkpoints/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml")
    cap = setup_video()
    
    ret, frame = cap.read()
    if not ret:
        return
    
    original_size = (frame.shape[1], frame.shape[0])
    resized_size = (frame.shape[1] // 2, frame.shape[0] // 2)
    frame_resized = cv2.resize(frame, resized_size)
    predictor.load_first_frame(frame_resized)
    
    obj_id = 1
    obj_ids = []
    points = [[175, 120], [175, 130], [160, 120], [190, 120]]
    labels = [1, 0, 0, 0]
    _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(0, obj_id, points, labels)
    obj_ids.append(obj_id)
    obj_id += 1
    
    frame_idx = 1
    fin_obj_id = int(obj_id)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % fin_obj_id == 1:
            frame_resized = cv2.resize(frame, resized_size)
            out_obj_ids, out_mask_logits = predictor.track(frame_resized)
            overlay = frame_resized.copy()
            
            colors = [(255, 0, 0, 100), (0, 255, 0, 100), (0, 0, 255, 100)]
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy()
                h, w = mask.shape[-2:]
                mask = mask.reshape(h, w).astype(np.uint8)
                color = colors[i % len(colors)]
                colored_mask = np.zeros_like(frame_resized, dtype=np.uint8)
                for j in range(3):
                    colored_mask[:, :, j] = np.where(mask == 1, color[j], 0)
                overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.6, 0)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = box.astype(np.int32)
                    angle = abs(rect[2])
                    if rect[1][0] < rect[1][1]:
                        angle = 90 + rect[2]
                    angle = abs(angle)
                    if angle > 90:
                        angle = 180 - angle
                    cv2.putText(overlay, f"Angle: {angle:.1f}", (box[0][0], box[0][1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 255, 20), 2, cv2.LINE_AA)
                
            _, buffer = cv2.imencode('.jpg', overlay)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        frame_idx += 1

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
