from scipy.spatial.transform import Rotation as R_scipy
import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
from torchvision.transforms import v2 as tf
from model import UNet
from data_process import transforms
import time
import torch

# -------------------- 分割模型 --------------------
class Segmentation():
    def __init__(self, device, check_point):
        OUT_CHANNELS = 5
        encoder_in_channel_list = [64, 128, 256]
        decoder_in_channel_list = [1024, 512, 256, 128]
        self.DEVICE = device
        self.model = UNet(3, encoder_in_channel_list, decoder_in_channel_list, OUT_CHANNELS, kernel_size=3, padding='same', bias=False).to(device)
        self.model.load_state_dict(torch.load(check_point, map_location=device))
        self.model.eval()
        self.transform = transforms['test']
        self.COLORMAP = {
            0: (0, 0, 0),
            1: (0, 255, 0),
            2: (255, 0, 0),
            3: (123, 112, 255),
            4: (255, 0, 0),
        }

    def label_to_rgb(self, mask_2d):
        h, w = mask_2d.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in self.COLORMAP.items():
            rgb[mask_2d == class_id] = color
        return rgb

    def predict(self, input_tensor, target_class=4):
        with torch.no_grad():
            pred = self.model(input_tensor)
            pred_mask = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            pred_mask_sea = np.where(pred_mask == target_class, 4, 0)
            pred_mask_rgb = self.label_to_rgb(pred_mask_sea)
        return pred_mask_rgb, pred_mask_sea

    def extract_sea_horizon(self, pred_mask, sea_class=4):
        height, width = pred_mask.shape
        horizon_y = np.full(width, -1)
        for x in range(width):
            sea_pixels = np.where(pred_mask[:, x] == sea_class)[0]
            if sea_pixels.size > 0:
                horizon_y[x] = sea_pixels.min()
        return horizon_y

# -------------------- 距離與速度估計 --------------------
class Speed_Estimation():
    def __init__(self, fx, fy, w, h, imu, horizon_y):
        self.K = np.array([[fx, 0, w // 2], [0, fy, h // 2], [0, 0, 1]], dtype=float)
        self.R = R_scipy.from_euler('xyz', imu, degrees=True).as_matrix()
        self.T = np.array([0, 0, 0], dtype=float)
        self.horizon_y = horizon_y
        self.D0 = 500

    def get_scale(self, u, v):
        if u < 0 or u >= len(self.horizon_y):
            return self.D0
        horizon_at_x = self.horizon_y[int(u)]
        if horizon_at_x <= 0:
            return self.D0

        fy_effective = self.K[1, 1]
        H_camera = 2.5

        delta_v = (v - horizon_at_x)
        if delta_v <= 0:
            delta_v = 1

        z = (H_camera * fy_effective) / delta_v
        z = np.clip(z, 0.5, 2000)
        return z

    def pixel_to_meter(self, u, v):
        z = self.get_scale(u, v)
        pixel_homog = np.array([u, v, 1], dtype=float)
        K_inv = np.linalg.inv(self.K)
        P_cam = z * (K_inv @ pixel_homog)
        P_world_rel = self.R @ P_cam + self.T
        return float(P_world_rel[0]), float(P_world_rel[2])  # X, Z

# -------------------- 主追蹤架構 --------------------
class Obj_Tracker():
    def __init__(self, target_classes, maxlen, cutpoint, path, check_point, seg_check_point, fx, fy, w, h, imu):
        self.target_classes = target_classes
        self.maxlen, self.cutpoint = maxlen, cutpoint
        self.path = path
        self.model = YOLO(check_point)
        self.seg = Segmentation(device="mps", check_point=seg_check_point)
        self.fx, self.fy, self.w, self.h = fx, fy, w, h
        self.imu = imu

    def _track(self):
        cap = cv2.VideoCapture(self.path)
        assert cap.isOpened()
        w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
        cap.release()

        video_writer = cv2.VideoWriter("tracked_with_speed.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        tracks = self.model.track(source=self.path, stream=True, persist=True, tracker="botsort.yaml")

        id_history, id_time = defaultdict(lambda: deque(maxlen=self.maxlen)), defaultdict(lambda: deque(maxlen=self.maxlen))

        for r in tracks:
            im0 = r.orig_img.copy()
            canvas = im0.copy()
            input_frame = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
            input_tensor = self.seg.transform(input_frame)
            input_tensor = tf.ToDtype(torch.float32, scale=True)(input_tensor).unsqueeze(0).to(self.seg.DEVICE)
            pred_mask_rgb, pred_mask = self.seg.predict(input_tensor)

            if pred_mask.shape[:2] != im0.shape[:2]:
                pred_mask_resized = cv2.resize(pred_mask, (im0.shape[1], im0.shape[0]), interpolation=cv2.INTER_NEAREST)
            else:
                pred_mask_resized = pred_mask

            horizon_y = self.seg.extract_sea_horizon(pred_mask_resized, sea_class=4)
            self.speed_estimator = Speed_Estimation(self.fx, self.fy, self.w, self.h, self.imu, horizon_y)

            for x in range(len(horizon_y)):
                y = horizon_y[x]
                if y >= 0:
                    cv2.circle(canvas, (x, y), 1, (0, 255, 255), -1)

            if r.boxes is None or r.boxes.id is None: continue

            for i in range(len(r.boxes.xyxy)):
                obj_id = int(r.boxes.id[i])
                x1, y1, x2, y2 = r.boxes.xyxy[i].tolist()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                id_history[obj_id].append((cx, cy))
                id_time[obj_id].append(time.time())
                cv2.circle(canvas, (cx, cy), 3, (0, 0, 255), -1)

            for obj_id in id_history:
                pts, ts = list(id_history[obj_id]), list(id_time[obj_id])
                if len(pts) >= self.cutpoint:
                    x_prev, y_prev = pts[-self.cutpoint]
                    x_curr, y_curr = pts[-1]
                    t_prev, t_curr = ts[-self.cutpoint], ts[-1]
                    dt = t_curr - t_prev
                    if dt <= 0: continue

                    X_prev, Z_prev = self.speed_estimator.pixel_to_meter(x_prev, y_prev)
                    X_curr, Z_curr = self.speed_estimator.pixel_to_meter(x_curr, y_curr)
                    distance_m = abs(Z_curr)
                    move_dist = np.hypot(X_curr - X_prev, Z_curr - Z_prev)
                    v_kmh = (move_dist / dt) * 3.6

                    cv2.putText(canvas, f"ID {obj_id} | {int(v_kmh)} km/h | Depth: {int(distance_m)} m", (int(x_curr), int(y_curr - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            video_writer.write(canvas)
            cv2.imshow("Tracked with Speed + Segmentation", canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        video_writer.release()
        cv2.destroyAllWindows()

# -------------------- 主程式入口 --------------------
if __name__ == "__main__":
    target_classes = ["ship"]
    maxlen, cutpoint = 64, 8
    path = "/Users/zhouchenghan/python/Full-shot/Unet/video/test_2.avi"
    check_point = "/Users/zhouchenghan/python/Full-shot/Ultralytics YOLO11.pt"
    seg_check_point = "/Users/zhouchenghan/python/Full-shot/Unet/early_stopped_model.pth"
    fx, fy, w, h = 5833, 5833, 1920, 1080
    imu = (0.0, 0.0, 0.0)

    tracker = Obj_Tracker(target_classes, maxlen, cutpoint, path, check_point, seg_check_point, fx, fy, w, h, imu)
    tracker._track()
    print("Tracking completed.")


# Single Image Monocular Horizon-based Range Estimation