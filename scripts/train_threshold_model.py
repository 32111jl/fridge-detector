import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import os
import json
from collections import defaultdict

class ThresholdTrainer:
    def __init__(self):
        # init mediapipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # collected measurements
        self.measurements = {
            'vertical': defaultdict(list),   # vertical hand measurements
            'sideways': defaultdict(list)    # sideways hand measurements
        }
        
        # current thresholds
        self.thresholds = {
            'vertical_hand': 0.15,
            'angle_bend': 50,
            'relative_occlusion': 0.45,
            'min_fingers': 3
        }
        
        # tracking variables
        self.hand_was_present = False
        self.trail_points = []
        self.max_trail_length = 20
        self.current_item = 1
        self.samples_per_item = 0
        
    def draw_status_overlay(self, frame):
        # main top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # time and camera
        time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(frame, time_str, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # collection status
        status = f"Item #{self.current_item} - Samples: {self.samples_per_item}"
        cv2.putText(frame, status, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # controls - centered
        controls = "SPACE: capture | N: next item | S: save"
        text_size = cv2.getTextSize(controls, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, controls, (text_x, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # quit instruction - right aligned
        quit_text = "Q: quit"
        quit_size = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(frame, quit_text, (frame.shape[1] - quit_size[0] - 20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def get_hand_measurements(self, hand_landmarks):
        """get all relevant hand measurements"""
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
        
        # get hand size
        hand_size = calculate_xz_distance(wrist, middle_mcp)
        
        # get vertical/sideways orientation measure
        hand_direction = abs(middle_mcp.y - wrist.y)
        
        # get finger angles
        angles = [
            calculate_angle(index_mcp, index_pip, index_tip),
            calculate_angle(middle_mcp, middle_pip, middle_tip),
            calculate_angle(ring_mcp, ring_pip, ring_tip),
            calculate_angle(pinky_mcp, pinky_pip, pinky_tip)
        ]
        
        # get finger-to-segment distances relative to hand size
        tip_to_segment_distances = [
            min(calculate_xz_distance(index_tip, index_pip), calculate_xz_distance(index_tip, index_mcp)),
            min(calculate_xz_distance(middle_tip, middle_pip), calculate_xz_distance(middle_tip, middle_mcp)),
            min(calculate_xz_distance(ring_tip, ring_pip), calculate_xz_distance(ring_tip, ring_mcp)),
            min(calculate_xz_distance(pinky_tip, pinky_pip), calculate_xz_distance(pinky_tip, pinky_mcp))
        ]
        relative_distances = [d / hand_size for d in tip_to_segment_distances]
        
        return {
            'hand_direction': hand_direction,
            'hand_size': hand_size,
            'angles': angles,
            'relative_distances': relative_distances
        }
    
    def analyze_measurements(self):
        """analyze collected measurements to determine optimal thresholds"""
        if not self.measurements['vertical'] and not self.measurements['sideways']:
            print("No measurements collected!")
            return
        
        # analyze vertical hand measurements
        if self.measurements['vertical']:
            angles = []
            for item_angles in self.measurements['vertical']['angles']:
                angles.extend(item_angles)
            if angles:
                # set angle threshold to capture 90% of measured angles
                self.thresholds['angle_bend'] = np.percentile(angles, 10)
        
        # analyze sideways hand measurements
        if self.measurements['sideways']:
            distances = []
            for item_distances in self.measurements['sideways']['relative_distances']:
                distances.extend(item_distances)
            if distances:
                # set occlusion threshold to capture 90% of measured distances
                self.thresholds['relative_occlusion'] = np.percentile(distances, 90)
        
        # set vertical hand threshold based on all measurements
        all_directions = (
            self.measurements['vertical']['hand_direction'] +
            self.measurements['sideways']['hand_direction']
        )
        if all_directions:
            self.thresholds['vertical_hand'] = np.percentile(all_directions, 50)
        
        print("\nAnalyzed measurements and updated thresholds:")
        for key, value in self.thresholds.items():
            print(f"{key}: {value:.3f}")
    
    def collect_measurements(self):
        """collect hand measurements from real grabs"""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("cannot open webcam")
        
        print("\nControls:")
        print("SPACE: Capture current hand position")
        print("N: Move to next item")
        print("S: Save and analyze measurements")
        print("Q: Quit\n")
        
        print("Hold an item and press SPACE to capture measurements")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # process frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # add trail effect
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                point = (int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0]))
                
                self.trail_points.append(point)
                if len(self.trail_points) > self.max_trail_length:
                    self.trail_points.pop(0)
                
                for i in range(1, len(self.trail_points)):
                    alpha = i / len(self.trail_points)
                    thickness = max(1, int(alpha * 10))
                    color = (
                        int(255 * (1 - alpha)),
                        int(255 * alpha),
                        255
                    )
                    cv2.line(frame, self.trail_points[i-1], self.trail_points[i], color, thickness)
                
                self.hand_was_present = True
            else:
                self.hand_was_present = False
                self.trail_points = []
            
            # draw ui
            frame = self.draw_status_overlay(frame)
            cv2.imshow('Measurement Collection', frame)
            
            # handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.analyze_measurements()
                self.save_thresholds()
                print("Measurements analyzed and thresholds saved!")
            elif key == ord('n'):
                self.current_item += 1
                self.samples_per_item = 0
                print(f"\nMoving to item #{self.current_item}")
                print("Hold the new item and press SPACE to capture measurements")
            elif key == 32:  # space
                if results.multi_hand_landmarks:
                    # get measurements
                    measurements = self.get_hand_measurements(hand_landmarks)
                    
                    # determine if vertical or sideways based on current threshold
                    if measurements['hand_direction'] > self.thresholds['vertical_hand']:
                        category = 'vertical'
                    else:
                        category = 'sideways'
                    
                    # store measurements
                    for key, value in measurements.items():
                        self.measurements[category][key].append(value)
                    
                    self.samples_per_item += 1
                    print(f"Captured {category} hand measurement (Sample {self.samples_per_item} for item {self.current_item})")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_thresholds(self):
        """save thresholds to file"""
        with open('hand_thresholds.json', 'w') as f:
            json.dump(self.thresholds, f, indent=4)
        print(f"Thresholds saved to hand_thresholds.json")
    
    def load_thresholds(self):
        """load thresholds from file"""
        try:
            with open('hand_thresholds.json', 'r') as f:
                self.thresholds = json.load(f)
            print("Loaded thresholds from hand_thresholds.json")
        except FileNotFoundError:
            print("No saved thresholds found, using defaults")

if __name__ == "__main__":
    trainer = ThresholdTrainer()
    trainer.load_thresholds()
    trainer.collect_measurements() 