import cv2 # type: ignore
import mediapipe as mp # type: ignore
import roboflow
from datetime import datetime
import numpy as np
import os
import shutil
import threading
from queue import Queue
from enum import Enum

class HandState(Enum):
    EMPTY = 0
    HOLDING = 1  # Simplified to just two states

class CameraState:
    def __init__(self):
        self.hand_state = HandState.EMPTY
        self.is_back = False
        self.hand_landmarks = None
        self.last_position = None
        self.last_seen_position = None
        self.stable_frames = 0
        self.holding_frames = 0  # Counter for consistent holding detection

class FruitDetection:
  def __init__(self, api_key):
    self.rf = roboflow.Roboflow(api_key)
    self.model = self.rf.workspace().project("fridge-detector-wdnfr").version("6").model
    self.model.confidence = 0.9
    self.model.overlap = 0.4
    self.last_pred = []
    self.last_time = datetime.now()
    
  def async_inference(self, frame):
    print("Starting inference...")
    def run_inference():
      temp_img_path = "temp.jpg"
      cv2.imwrite(temp_img_path, frame)
      prediction = self.model.predict(temp_img_path)
      print(f"Prediction: {prediction.json()}")
      self.last_pred = prediction.json()["predictions"]
      self.last_time = datetime.now()
    
    thread = threading.Thread(target=run_inference)
    thread.start()
  
  def draw_predictions(self, frame):
    if self.last_pred:
      print("Drawing predictions...")
      for prediction in self.last_pred:
        class_name = prediction["class"]
        confidence = prediction["confidence"]
        
        x_center = prediction["x"]
        y_center = prediction["y"]
        width  = prediction["width"]
        height = prediction["height"]
        # x1 = int((x_center - width/2) * frame.shape[1])
        # y1 = int((y_center - height/2) * frame.shape[0])
        # x2 = int((x_center + width/2) * frame.shape[1])
        # y2 = int((y_center + height/2) * frame.shape[0])
        x1 = int(x_center - width / 3)
        y1 = int(y_center - height / 2.5)
        x2 = int(x_center + width / 3)
        y2 = int(y_center + height / 2.5)
        
        print(f"Detected {class_name} at ({x_center}, {y_center}) with confidence {confidence}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} ({confidence:.2f})", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    return frame

class SimpleHandTracker:
    def __init__(self):
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
        
        # Set maximum FPS
        self.cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Use MJPG format for higher FPS
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer delay
        
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Store states
        self.hand_was_present = False
        self.holding_start_time = None
        self.grab_confidence_counter = 0
        self.confident_grab = False
        self.action_completed = False
        self.last_hand_position = None
        self.initial_hand_position = None
        
        # Visual effect variables
        self.trail_points = []
        self.max_trail_length = 20
        self.last_action = None
        self.last_action_time = None
        
        self.fruit_detector = FruitDetection("OaxF2iDz0uE7kTh60Odx")
        
    def is_grabbing(self, hand_landmarks):
        """check if hand is holding"""
        # get landmarks
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        def calculate_angle(p1, p2, p3):
            """get angle between 3 points"""
            v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
            
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            
            if v1_norm == 0 or v2_norm == 0:
                return 0
            
            v1 = v1 / v1_norm
            v2 = v2 / v2_norm
            
            angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
            return np.degrees(angle)
        
        def calculate_xz_distance(p1, p2):
            """get distance ignoring y axis"""
            return ((p1.x - p2.x)**2 + (p1.z - p2.z)**2)**0.5
        
        # get hand size (distance from wrist to middle finger mcp)
        hand_size = calculate_xz_distance(wrist, middle_mcp)
        
        # check hand orientation
        hand_direction = abs(middle_mcp.y - wrist.y)
        is_vertical = hand_direction > 0.15
        
        if is_vertical:
            # check finger bending
            angles = [
                calculate_angle(index_mcp, index_pip, index_tip),
                calculate_angle(middle_mcp, middle_pip, middle_tip),
                calculate_angle(ring_mcp, ring_pip, ring_tip),
                calculate_angle(pinky_mcp, pinky_pip, pinky_tip)
            ]
            
            angle_threshold = 50
            fingers_bent = sum(angle > angle_threshold for angle in angles)
            return fingers_bent >= 3
            
        else:
            # check finger occlusion relative to hand size
            tip_to_segment_distances = [
                min(calculate_xz_distance(index_tip, index_pip), calculate_xz_distance(index_tip, index_mcp)),
                min(calculate_xz_distance(middle_tip, middle_pip), calculate_xz_distance(middle_tip, middle_mcp)),
                min(calculate_xz_distance(ring_tip, ring_pip), calculate_xz_distance(ring_tip, ring_mcp)),
                min(calculate_xz_distance(pinky_tip, pinky_pip), calculate_xz_distance(pinky_tip, pinky_mcp))
            ]
            
            # normalize distances by hand size
            relative_distances = [dist / hand_size for dist in tip_to_segment_distances]
            
            # relative threshold (proportion of hand size)
            relative_threshold = 0.45  # increased from 0.3 to be more sensitive
            occluded_fingers = sum(dist < relative_threshold for dist in relative_distances)
            return occluded_fingers >= 3
        
    def add_visual_effects(self, frame, hand_landmarks):
        # Add glow effect
        overlay = frame.copy()
        for landmark in hand_landmarks.landmark:
            center = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
            color = (0, 255, 0) if self.confident_grab else (0, 165, 255)
            cv2.circle(overlay, center, 20, color, -1)
        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
        
        # Add trail effect
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        point = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
        
        self.trail_points.append(point)
        if len(self.trail_points) > self.max_trail_length:
            self.trail_points.pop(0)
            
        for i in range(1, len(self.trail_points)):
            alpha = i / len(self.trail_points)
            thickness = max(1, int(alpha * 10))  # Ensure thickness is at least 1
            color = (
                int(255 * (1 - alpha)),  # B
                int(255 * alpha),        # G
                255                      # R
            )
            cv2.line(frame, self.trail_points[i-1], self.trail_points[i], color, thickness)
        
        return frame
        
    def draw_status_overlay(self, frame):
        # main top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # time and camera
        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(frame, time_str, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "CAM1", (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # status and confidence
        status = "HOLDING" if self.confident_grab else "NOT HOLDING"
        color = (0, 255, 0) if self.confident_grab else (0, 165, 255)
        text_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, status, (text_x, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # hand orientation
        if hasattr(self, 'hand_orientation'):
            orientation_text = f"Hand: {self.hand_orientation}"
            cv2.putText(frame, orientation_text, (text_x + text_size[0] + 20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # confidence bar
        bar_length = 200
        bar_height = 15
        filled_length = int(bar_length * (min(self.grab_confidence_counter, 10) / 10))
        bar_x = (frame.shape[1] - bar_length) // 2
        cv2.rectangle(frame, (bar_x, 45), (bar_x + bar_length, 45 + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, 45), (bar_x + filled_length, 45 + bar_height), color, -1)
        
        # confidence value
        conf_text = f"{self.grab_confidence_counter}/10"
        conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        conf_x = bar_x + (bar_length - conf_size[0]) // 2
        cv2.putText(frame, conf_text, (conf_x, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # hold duration
        if self.confident_grab and self.holding_start_time is not None:
            duration = (current_time - self.holding_start_time).total_seconds()
            duration_text = f"Hold: {duration:.1f}s"
            cv2.putText(frame, duration_text, (frame.shape[1] - 200, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # last action with fade
        if self.last_action and self.last_action_time:
            time_since_action = (current_time - self.last_action_time).total_seconds()
            if time_since_action < 2.0:
                alpha = 1.0 - (time_since_action / 2.0)
                action_color = (0, int(255 * alpha), int(255 * alpha))
                cv2.putText(frame, f"Last Action: {self.last_action}", (frame.shape[1] - 300, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)
        
        return frame
        
    def run(self):
        print("Press 'q' to quit")
        print("Left side = PUT IN, Right side = TAKE OUT")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # check palm/back
                    self.hand_orientation = self.detect_palm_or_back(hand_landmarks)
                    
                    # add effects
                    frame = self.add_visual_effects(frame, hand_landmarks)
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # check grab
                    current_grab = self.is_grabbing(hand_landmarks)
                    
                    if current_grab:
                        self.grab_confidence_counter = min(10, self.grab_confidence_counter + 1)
                        if self.grab_confidence_counter >= 5: # start holding
                            if not self.confident_grab:
                                self.holding_start_time = datetime.now()
                                # predictions = self.fruit_detector.async_inference(frame)
                            # frame = self.fruit_detector.draw_predictions(frame, predictions)
                            self.fruit_detector.async_inference(frame)
                            self.confident_grab = True
                    else:
                        self.grab_confidence_counter = max(0, self.grab_confidence_counter - 1)
                        if self.grab_confidence_counter < 3: # reset grab state/start time
                            self.confident_grab = False
                            self.holding_start_time = None
                    
                    # update position
                    self.last_hand_position = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
                    
                    # position marker
                    pos_x = int(self.last_hand_position * frame.shape[1])
                    cv2.line(frame, (pos_x, frame.shape[0]), (pos_x, frame.shape[0]-50), 
                            (0, 255, 255), 2)
                    cv2.circle(frame, (pos_x, frame.shape[0]-25), 5, (0, 255, 255), -1)
                    
                    self.hand_was_present = True
            
            else:
                # check action on hand lost
                if (self.hand_was_present and self.confident_grab and 
                    self.last_hand_position is not None and 
                    self.holding_start_time is not None):
                    # get action from position
                    center_x = 0.5
                    if abs(self.last_hand_position - center_x) > 0.2:
                        if self.last_hand_position > center_x:
                            action = "TAKE OUT"
                        else:
                            action = "PUT IN"
                        
                        hold_duration = (datetime.now() - self.holding_start_time).total_seconds()
                        print(f"Action: {action} (Position: {self.last_hand_position:.2f}, Duration: {hold_duration:.1f}s)")
                        
                        self.last_action = action
                        self.last_action_time = datetime.now()
                
                # reset states
                self.hand_was_present = False
                self.confident_grab = False
                self.holding_start_time = None
                self.grab_confidence_counter = 0
                self.trail_points = []
            
            frame = self.fruit_detector.draw_predictions(frame)
            
            # center line
            mid_x = int(frame.shape[1]/2)
            cv2.line(frame, (mid_x, 0), (mid_x, frame.shape[0]), 
                    (100, 100, 100), 1)
            
            # zone labels
            label_y = frame.shape[0]-30
            label_bg_alpha = 0.7
            
            # left zone
            left_overlay = frame.copy()
            cv2.rectangle(left_overlay, (30, label_y-25), (150, label_y+10), (0, 0, 0), -1)
            frame = cv2.addWeighted(left_overlay, label_bg_alpha, frame, 1 - label_bg_alpha, 0)
            cv2.putText(frame, "PUT IN", (50, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # right zone
            right_overlay = frame.copy()
            cv2.rectangle(right_overlay, (frame.shape[1]-170, label_y-25), 
                          (frame.shape[1]-30, label_y+10), (0, 0, 0), -1)
            frame = cv2.addWeighted(right_overlay, label_bg_alpha, frame, 1 - label_bg_alpha, 0)
            cv2.putText(frame, "TAKE OUT", (frame.shape[1]-150, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # status overlay
            frame = self.draw_status_overlay(frame)
            
            cv2.imshow('Hand Tracking', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

    def detect_palm_or_back(self, hand_landmarks):
        """check palm vs back of hand"""
        # fingertip landmarks
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # palm landmarks
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        pinky_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        
        # avg z positions
        fingertip_z = (index_tip.z + middle_tip.z + ring_tip.z + pinky_tip.z) / 4
        palm_z = (index_mcp.z + pinky_mcp.z + wrist.z) / 3
        
        # closer to camera = palm
        z_diff = fingertip_z - palm_z
        return "palm" if z_diff < 0 else "back"

class EnhancedHandTracker(SimpleHandTracker):
    def __init__(self):
        super().__init__()
        self.trail_points = []
        self.max_trail_length = 20
        self.overlay = None
        self.last_action_time = None
        self.last_action = None
        self.action_display_duration = 2.0  # seconds

class DualCameraTracker(SimpleHandTracker):
    def __init__(self):
        super().__init__()
        # Disable CAM2 initialization
        self.cap2 = None
        
        # Initialize only CAM1 state
        self.cam1_state = CameraState()
        
        # Initialize action states
        self.current_action = None
        self.last_capture_time = None
        self.capture_cooldown = 0.5  # seconds between captures
        
        # Initialize trail points for CAM1 only
        self.trail_points_cam1 = []
        self.max_trail_length = 20
        
        # Clean and setup directories
        self.base_dir = "captured_items"
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        
        self.taking_dir = os.path.join(self.base_dir, "taking_out")
        self.putting_dir = os.path.join(self.base_dir, "putting_in")
        for directory in [self.taking_dir, self.putting_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Add tracking for holding state
        self.cam1_was_holding = False
        self.cam2_was_holding = False
        self.last_capture_time = None
        self.capture_cooldown = 0.5  # seconds
        
        # Add sequence tracking
        self.current_sequence = []
        self.sequence_start_time = None
        self.capture_cooldown = 0.1  # Reduce cooldown for more frequent captures
        self.current_sequence_dir = None

if __name__ == "__main__":
    tracker = SimpleHandTracker()
    tracker.run()