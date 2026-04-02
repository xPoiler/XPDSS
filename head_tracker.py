import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading
import time
import numpy as np
import os
import urllib.request

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

class HeadTracker:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        
        # Download model if not exists
        if not os.path.exists(MODEL_PATH):
            print("Downloading Face Landmarker model...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=4,
            min_face_detection_confidence=0.10,  # Lowered for wide-angle stickiness
            min_face_presence_confidence=0.10,
            min_tracking_confidence=0.10
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        
        # Smoothed Output State
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.pos_z = 1.0
        self.rot_yaw = 0.0  # Radians
        
        self.smoothing = 0.35 # Faster responsiveness
        self._valid_frames = 0
        
        self.running = False
        self.show_preview = False
        self._window_visible = False
        self.thread = None
        self.frame_count = 0
        self.fps = 0
        
    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        print(f"HeadTracker: Started on camera {self.camera_index}")
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if hasattr(self, 'detector'):
            try:
                self.detector.close()
            except Exception:
                pass
        print("HeadTracker: Stopped")
            
    def _worker(self):
        last_time = time.time()
        cap = None
        consecutive_failures = 0
        max_failures = 30 # At 30fps, ~1 second of failure before reconnecting
        
        while self.running:
            raw_yaw = 0.0 # reset for this frame
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                import sys
                if sys.platform == 'win32':
                    cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(self.camera_index)
                
                # --- Force Higher Resolution (Enables detection of smaller face at distance) ---
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                if not cap.isOpened():
                    print(f"HeadTracker: Waiting for camera {self.camera_index}...")
                    time.sleep(2.0)
                    continue
                else:
                    print(f"HeadTracker: Connected to camera {self.camera_index}")
                    consecutive_failures = 0

            success, image = cap.read()
            if not success:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                     print(f"HeadTracker Warning: Frame grab failed {max_failures} times. Reconnecting...")
                     if cap is not None:
                          cap.release()
                     cap = None
                     time.sleep(1.0)
                else:
                     time.sleep(0.01)
                continue
                
            consecutive_failures = 0 # Reset on success

                
            self.frame_count += 1
            if time.time() - last_time > 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                last_time = time.time()
                
            image = cv2.flip(image, 1)
            h, w, _ = image.shape

            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            try:
                detection_result = self.detector.detect(mp_image)
            except Exception as e:
                print(f"Detector crashed error: {e}")
                time.sleep(0.1)
                continue
            
            # --- Extract Rotation ---
            if hasattr(detection_result, 'facial_transformation_matrixes') and detection_result.facial_transformation_matrixes:
                try:
                    m = detection_result.facial_transformation_matrixes[0]
                    arr = np.array(m.data).reshape(4, 4)
                    raw_yaw = np.arctan2(arr[0, 2], arr[0, 0])
                    yaw_speed = np.abs(raw_yaw - self.rot_yaw)
                    alpha_yaw = min(0.90, self.smoothing + max(0.0, 0.20 - yaw_speed * 10.0))
                    self.rot_yaw = self.rot_yaw * alpha_yaw + raw_yaw * (1 - alpha_yaw)
                except Exception:
                    pass

            if detection_result.face_landmarks:
                sum_x, sum_y, sum_z = 0.0, 0.0, 0.0
                valid_count = 0
                valid_coords = [] # Deduplicate overlapping face bounding boxes
                
                h, w, c = image.shape
                valid_frames = getattr(self, '_valid_frames', 0)
                
                for landmarks in detection_result.face_landmarks:
                    nose = landmarks[4]
                    eye_l = landmarks[33]
                    eye_r = landmarks[263]
                    
                    # Absolute Pixels based scaling
                    dx = (eye_r.x - eye_l.x) * w 
                    dy = (eye_r.y - eye_l.y) * h 
                    
                    cosine_yaw = np.cos(getattr(self, 'rot_yaw', 0.0))
                    cosine_yaw = max(0.4, abs(cosine_yaw))
                    corrected_dx = dx / cosine_yaw
                    
                    eye_dist_px = np.sqrt(corrected_dx**2 + dy**2)
                    eye_dist = eye_dist_px / w # Normalized to full frame width
                    
                    # 1. Size Gating: Reject absolute tiny faces 
                    if eye_dist < 0.012: 
                        continue
                        
                    raw_z = 0.17 / (eye_dist + 1e-6)
                    
                    raw_x = nose.x - 0.5
                    raw_y = nose.y - 0.5
                    
                    # 2. Spatial gating: Reject if too far from last valid position (background clutter)
                    dist_to_last = np.sqrt((raw_x - self.pos_x)**2 + (raw_y - self.pos_y)**2)
                    if valid_frames > 8 and dist_to_last > 0.45 and eye_dist < 0.045:
                        continue # Skip false face in background
                        
                    # 3. Overlap Deduplication: Reject if 2 faces occupy same space (sensitivity duplicate)
                    is_dup = False
                    for (vx, vy) in valid_coords:
                        if np.sqrt((raw_x - vx)**2 + (raw_y - vy)**2) < 0.15:
                            is_dup = True
                            break
                    if is_dup:
                        continue # Skip duplicate
                        
                    sum_x += raw_x
                    sum_y += raw_y
                    sum_z += raw_z
                    valid_coords.append((raw_x, raw_y))
                    valid_count += 1
                    
                    for node in [nose, eye_l, eye_r]:
                        px = int(node.x * w)
                        py = int(node.y * h) 
                        cv2.circle(image, (px, py), 4, (0, 0, 255), -1)
                        
                if valid_count > 0:
                    avg_x = sum_x / valid_count
                    avg_y = sum_y / valid_count
                    avg_z = sum_z / valid_count
                    self._valid_frames = valid_frames + 1
                    setattr(self, '_miss_count', 0)
                else:
                    self._valid_frames = max(0, valid_frames - 1)
                    miss_count = getattr(self, '_miss_count', 0) + 1
                    setattr(self, '_miss_count', miss_count)
                    if miss_count > 6: # Unlock spatial gate if consecutive skips exceed 0.2s
                        self._valid_frames = 0
                    continue # Skip update for this frame group if everything filtered
                
                # Adaptive smoothing for lock and fast response
                speed = np.sqrt((avg_x - self.pos_x)**2 + (avg_y - self.pos_y)**2)
                alpha_dyn = min(0.92, self.smoothing + max(0.0, 0.22 - speed * 10.0))
                
                self.pos_x = self.pos_x * alpha_dyn + avg_x * (1 - alpha_dyn)
                self.pos_y = self.pos_y * alpha_dyn + avg_y * (1 - alpha_dyn)
                self.pos_z = self.pos_z * alpha_dyn + avg_z * (1 - alpha_dyn)
                    
            if self.show_preview:
                cv2.imshow("XTC 3D Head Tracker", image)
                self._window_visible = True
                if cv2.waitKey(1) & 0xFF == 27: # ESC to stop
                     self.show_preview = False
                     try:
                         cv2.destroyWindow("XTC 3D Head Tracker")
                     except Exception:
                         pass
                     self._window_visible = False
                else:
                    # Detect Window Close ('X' button in title bar)
                    try:
                         if cv2.getWindowProperty("XTC 3D Head Tracker", cv2.WND_PROP_VISIBLE) < 1:
                              self.show_preview = False
                              try:
                                  cv2.destroyWindow("XTC 3D Head Tracker")
                              except Exception:
                                  pass
                              self._window_visible = False
                    except Exception:
                         pass
            else:
                if self._window_visible:
                     try:
                         cv2.destroyWindow("XTC 3D Head Tracker")
                     except Exception:
                         pass
                     self._window_visible = False
                     
        if cap is not None:
            cap.release()

        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="Camera Index")
    args = parser.parse_args()
    
    tracker = HeadTracker(camera_index=args.cam)
    tracker.start()
    
    print("\n--- HeadTracker Console View ---")
    print("Press Ctrl+C to exit.\n")
    try:
        while True:
            hbar = int((tracker.pos_x + 0.5) * 40)
            hbar = max(0, min(39, hbar))
            bar_str = "[" + "-" * hbar + "|" + "-" * (39 - hbar) + "]"
            yaw_deg = np.degrees(tracker.rot_yaw)
            print(f"X: {tracker.pos_x:+0.3f} {bar_str} | Z: {tracker.pos_z:0.3f} | Yaw: {yaw_deg:+.1f}° | FPS: {tracker.fps}", end="\r")
            time.sleep(0.1)
    except KeyboardInterrupt:
         tracker.stop()
         print("\nExit.")
