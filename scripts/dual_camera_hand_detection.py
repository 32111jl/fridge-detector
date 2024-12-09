import cv2
import mediapipe as mp
import numpy as np
import os
import shutil
from datetime import datetime
import time
from enum import Enum
import threading
from queue import Queue
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque

class HandState(Enum):
    EMPTY = 0
    HOLDING = 1

class HandSide(Enum):
    PALM = 0
    BACK = 1
    UNKNOWN = 2

class CameraState:
    def __init__(self):
        self.hand_landmarks = None
        self.hand_state = HandState.EMPTY
        self.hand_side = HandSide.UNKNOWN
        self.holding_start_time = None

class CameraThread(threading.Thread):
    def __init__(self, camera_id, name, frame_queue):
        super().__init__()
        self.camera_id = camera_id
        self.name = name
        self.frame_queue = frame_queue
        self.stopped = False
        self.cap = None
        
    def run(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open {self.name}")
            return
            
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put((ret, frame))
            else:
                break
                
        if self.cap:
            self.cap.release()

class DualCameraHandDetector:
    def __init__(self):
        print("\nInitializing dual camera hand detection system...")
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        
        # Release any existing cameras
        for i in range(10):
            try:
                cap = cv2.VideoCapture(i)
                cap.release()
            except:
                pass
                
        time.sleep(1)  # Wait for cameras to release
        
        # Initialize camera threads
        self.frame_queue1 = Queue(maxsize=2)  # Increased queue size
        self.frame_queue2 = Queue(maxsize=2)  # Increased queue size
        
        self.cam1_thread = CameraThread(0, "External Camera", self.frame_queue1)
        self.cam2_thread = CameraThread(1, "Built-in Camera", self.frame_queue2)
        
        # Initialize hand detection with higher confidence for built-in camera
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        
        # Create separate hand detectors for each camera
        self.hands1 = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        )
        self.hands2 = self.mp_hands.Hands(  # Built-in camera detector with lower confidence threshold
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.3  # Lower threshold for built-in camera
        )
        
        # Initialize states
        self.cam1_state = CameraState()
        self.cam2_state = CameraState()
        
        # Initialize trail effects
        self.trail_points_cam1 = []
        self.trail_points_cam2 = []
        self.max_trail_length = 20
        
        # Action states
        self.current_action = None
        self.is_recording = False
        self.recording_start_time = None
        self.current_sequence_dir = None
        self.frame_count = 0
        self.palm_camera = None
        
        # Setup capture directories
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_dir = f"captured_items_{self.timestamp}"
        if os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir)
        
        # Create directories for different types of captures
        self.taking_dir = os.path.join(self.base_dir, "taking_out")
        self.putting_dir = os.path.join(self.base_dir, "putting_in")
        
        # Create subdirectories for raw and interface images
        for main_dir in [self.taking_dir, self.putting_dir]:
            for sub_dir in ["raw", "interface", "cropped"]:
                for view in ["palm", "back"]:
                    os.makedirs(os.path.join(main_dir, sub_dir, view), exist_ok=True)
            
        # Add tracking for last known positions
        self.last_palm_x = None
        self.action_zone = None
        
        # Add performance tracking
        self.mp_times = deque(maxlen=1000)
        self.mp_confidences = deque(maxlen=1000)
        self.yolo_times = deque(maxlen=1000)
        self.yolo_confidences = deque(maxlen=1000)
        self.landmark_confidences = {
            'MediaPipe': np.zeros((21, 1)),  # 21 landmarks
            'YOLO': np.zeros((21, 1))
        }
        self.frame_count = 0
        
    def is_grabbing(self, hand_landmarks):
        """Check if hand is in grabbing position using relative distances"""
        # Get fingertip and mcp landmarks
        fingertips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        ]
        
        mcps = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_CMC],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_MCP]
        ]
        
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        curled_fingers = 0
        
        # Get hand size (distance from wrist to middle finger MCP) for normalization
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        hand_size = ((middle_mcp.x - wrist.x) ** 2 + (middle_mcp.y - wrist.y) ** 2) ** 0.5
        
        # Fine-tune sensitivity for both curl and occlusion
        CURL_THRESHOLD = 1.8  # Increased from 1.7 to make curl more sensitive
        OCCLUSION_THRESHOLD = 0.85  # Decreased from 0.9 to make occlusion more sensitive
        
        for tip, mcp in zip(fingertips, mcps):
            # Calculate relative distances (normalized by hand size)
            tip_to_wrist = ((tip.x - wrist.x) ** 2 + (tip.y - wrist.y) ** 2) ** 0.5 / hand_size
            mcp_to_wrist = ((mcp.x - wrist.x) ** 2 + (mcp.y - wrist.y) ** 2) ** 0.5 / hand_size
            
            # Check both curling and occlusion
            is_curled = tip_to_wrist < mcp_to_wrist * CURL_THRESHOLD
            is_occluded = tip_to_wrist < mcp_to_wrist * OCCLUSION_THRESHOLD
            
            if is_curled or is_occluded:
                curled_fingers += 1
        
        return curled_fingers >= 3  # Still require 3 fingers
        
    def detect_hand_side(self, hand_landmarks, is_mirrored=False):
        """
        detect palm/back side using finger depth while grabbing:
        - palm: finger tips closer to camera than knuckles
        - back: finger tips further from camera than knuckles
        """
        # get finger tips and their mcp (knuckle) positions
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        
        index_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_MCP]
        middle_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
        ring_mcp = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_MCP]
        
        # check if tips are closer to camera than mcps
        # negative z means closer to camera
        index_closer = index_tip.z < index_mcp.z
        middle_closer = middle_tip.z < middle_mcp.z
        ring_closer = ring_tip.z < ring_mcp.z
        
        # count fingers closer to camera
        fingers_closer = sum([index_closer, middle_closer, ring_closer])
        
        # palm if majority of fingers are closer
        is_palm = fingers_closer >= 2
        
        return HandSide.PALM if is_palm else HandSide.BACK
        
    def process_frame(self, frame, camera_state, trail_points, camera_name, hand_detector, is_mirrored=False):
        if frame is None:
            return None
            
        # Mirror the frame first if needed, before processing
        if is_mirrored:
            frame = cv2.flip(frame, 1)
            
        # Start timing
        start_time = time.time()
        
        # Process frame with MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(rgb_frame)
        
        # Record MediaPipe processing time
        mp_process_time = (time.time() - start_time) * 1000  # Convert to ms
        self.mp_times.append(mp_process_time)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Record confidence scores for each landmark
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    self.landmark_confidences['MediaPipe'][idx] += landmark.visibility
                
                # Record overall confidence
                avg_confidence = np.mean([lm.visibility for lm in hand_landmarks.landmark])
                self.mp_confidences.append(avg_confidence)
        
        # add zone markers
        height, width = frame.shape[:2]
        
        # taking out zone (left side)
        cv2.rectangle(frame, (0, 0), (int(width * 0.4), height), (0, 255, 0), 2)
        cv2.putText(frame, "take out", (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # putting in zone (right side)
        cv2.rectangle(frame, (int(width * 0.6), 0), (width, height), (0, 165, 255), 2)
        cv2.putText(frame, "put in", (width - 200, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        
        # add bottom marker
        cv2.rectangle(frame, (0, height - 5), (width, height), (0, 255, 255), -1)
        
        # process frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(rgb_frame)
        
        # reset if no hand detected
        if not results.multi_hand_landmarks:
            camera_state.hand_landmarks = None
            camera_state.hand_state = HandState.EMPTY
            camera_state.holding_start_time = None
            # only reset side if not recording
            if not self.is_recording:
                camera_state.hand_side = HandSide.UNKNOWN
                if self.palm_camera == camera_name:
                    self.palm_camera = None
            frame = self.draw_status_overlay(frame, camera_state, camera_name)
            return frame
        
        # process detected hand
        for hand_landmarks in results.multi_hand_landmarks:
            # update hand state
            camera_state.hand_landmarks = hand_landmarks
            was_not_holding = camera_state.hand_state != HandState.HOLDING
            camera_state.hand_state = HandState.HOLDING if self.is_grabbing(hand_landmarks) else HandState.EMPTY
            
            # reset sides if just started holding
            if was_not_holding and camera_state.hand_state == HandState.HOLDING:
                self.palm_camera = None
                camera_state.hand_side = HandSide.UNKNOWN
                if "External" in camera_name:
                    self.cam2_state.hand_side = HandSide.UNKNOWN
                else:
                    self.cam1_state.hand_side = HandSide.UNKNOWN
            
            # detect side if holding and not determined yet
            if camera_state.hand_state == HandState.HOLDING and camera_state.hand_side == HandSide.UNKNOWN:
                detected_side = self.detect_hand_side(hand_landmarks, is_mirrored)
                
                # if no palm camera set yet
                if self.palm_camera is None:
                    # set as palm camera if palm detected
                    if detected_side == HandSide.PALM:
                        self.palm_camera = camera_name
                        camera_state.hand_side = HandSide.PALM
                        # set other camera to back
                        if "External" in camera_name:
                            self.cam2_state.hand_side = HandSide.BACK
                        else:
                            self.cam1_state.hand_side = HandSide.BACK
                    else:
                        # wait for other camera if back detected
                        camera_state.hand_side = HandSide.UNKNOWN
                else:
                    # use existing palm camera assignment
                    camera_state.hand_side = HandSide.PALM if camera_name == self.palm_camera else HandSide.BACK
            
            # get wrist position for marker and trail
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            marker_x = int(wrist.x * frame.shape[1])
            marker_y = int(wrist.y * frame.shape[0])
            
            # add yellow position marker with black outline
            cv2.circle(frame, (marker_x, marker_y), 16, (0, 0, 0), -1)  # Black outline
            cv2.circle(frame, (marker_x, marker_y), 12, (0, 255, 255), -1)  # Yellow marker
            
            # add trail effect
            point = (marker_x, marker_y)
            trail_points.append(point)
            if len(trail_points) > self.max_trail_length:
                trail_points.pop(0)
                
            for i in range(1, len(trail_points)):
                alpha = i / len(trail_points)
                thickness = max(1, int(alpha * 10))
                color = (
                    int(255 * (1 - alpha)),
                    int(255 * alpha),
                    255
                )
                cv2.line(frame, trail_points[i-1], trail_points[i], color, thickness)
            
            # draw hand landmarks with glow effect
            overlay = frame.copy()
            for connection in self.mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                
                start_pos = hand_landmarks.landmark[start_idx]
                end_pos = hand_landmarks.landmark[end_idx]
                
                start_pos = (int(start_pos.x * frame.shape[1]), int(start_pos.y * frame.shape[0]))
                end_pos = (int(end_pos.x * frame.shape[1]), int(end_pos.y * frame.shape[0]))
                
                # draw thicker lines for glow effect
                color = (0, 255, 0) if self.is_grabbing(hand_landmarks) else (0, 165, 255)
                cv2.line(overlay, start_pos, end_pos, color, 8)
                
            # add the glow effect
            frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
            
            # draw the actual landmarks
            self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            # update holding time
            if camera_state.hand_state == HandState.HOLDING:
                if camera_state.holding_start_time is None:
                    camera_state.holding_start_time = time.time()
            else:
                camera_state.holding_start_time = None
        
        # add status overlay
        frame = self.draw_status_overlay(frame, camera_state, camera_name)
        return frame
        
    def draw_status_overlay(self, frame, camera_state, camera_name):
        # Main top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Time and camera label
        current_time = datetime.now()
        time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        cv2.putText(frame, time_str, (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, camera_name, (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Status - always show status, even when no hand detected
        text_x = frame.shape[1] // 2 - 100  # Center position
        
        if camera_state.hand_landmarks is not None:
            status = "HOLDING" if camera_state.hand_state == HandState.HOLDING else "NOT HOLDING"
            color = (0, 255, 0) if camera_state.hand_state == HandState.HOLDING else (0, 165, 255)
        else:
            status = "NO HAND"
            color = (0, 165, 255)
            
        cv2.putText(frame, status, (text_x, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        # Add hand side - always show side, even when no hand detected
        side_str = f"Side: {camera_state.hand_side.name}"
        cv2.putText(frame, side_str, (text_x, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show duration only when holding
        if camera_state.holding_start_time is not None:
            duration = time.time() - camera_state.holding_start_time
            duration_str = f"Duration: {duration:.1f}s"
            cv2.putText(frame, duration_str, (frame.shape[1] - 200, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        return frame
        
    def check_action(self):
        # Get current hand position from external camera only
        if self.cam1_state.hand_landmarks:
            hand_x = self.cam1_state.hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST].x
            
            # Check if hand is holding and sides are correct (opposite for each camera)
            is_holding = self.cam1_state.hand_state == HandState.HOLDING
            sides_correct = (
                (self.cam1_state.hand_side == HandSide.PALM and self.cam2_state.hand_side == HandSide.BACK) or
                (self.cam1_state.hand_side == HandSide.BACK and self.cam2_state.hand_side == HandSide.PALM)
            )
            
            if is_holding and sides_correct:
                # Start recording if not already recording
                if not self.is_recording:
                    self.start_recording()
                
                # Update last known position
                self.last_palm_x = hand_x
                return None  # No action while hand is visible
                
        else:
            # Hand disappeared - check if we should trigger action
            if self.is_recording and self.last_palm_x is not None:
                # Determine action based on where the hand disappeared
                if self.last_palm_x < 0.4:
                    self.action_zone = "taking_out"
                elif self.last_palm_x > 0.6:
                    self.action_zone = "putting_in"
                else:
                    self.action_zone = None
                
                if self.action_zone:
                    action = self.action_zone
                    # Reset all hand states before stopping recording
                    self.cam1_state.hand_side = HandSide.UNKNOWN
                    self.cam2_state.hand_side = HandSide.UNKNOWN
                    self.palm_camera = None
                    # Stop recording
                    self.stop_recording()
                    # Reset tracking
                    self.last_palm_x = None
                    self.action_zone = None
                    return action
                
        return None
        
    def start_recording(self):
        """Start recording a new sequence"""
        self.is_recording = True
        self.recording_start_time = time.time()
        self.frame_count = 0
        
        # Create sequence timestamp
        self.sequence_time = datetime.now().strftime("%H%M%S_%f")[:-3]
        
        # Create temporary directory for recording
        self.current_sequence_dir = os.path.join(self.base_dir, "temp_sequence", self.sequence_time)
        
        # Create subdirectories for raw, interface, and cropped images
        for sub_dir in ["raw", "interface", "cropped"]:
            for view in ["palm", "back"]:
                os.makedirs(os.path.join(self.current_sequence_dir, sub_dir, view), exist_ok=True)
        
        print(f"\nStarted recording sequence at {self.sequence_time}")

    def get_hand_crop(self, frame, hand_landmarks, padding=50):
        """Get cropped image ensuring hand is included and detecting item corners around it"""
        frame_height, frame_width = frame.shape[:2]
        
        # First get the hand region
        hand_x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
        hand_y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]
        
        # Get hand bounding box with small padding
        hand_padding = 20
        hand_x_min = max(0, int(min(hand_x_coords) - hand_padding))
        hand_y_min = max(0, int(min(hand_y_coords) - hand_padding))
        hand_x_max = min(frame_width, int(max(hand_x_coords) + hand_padding))
        hand_y_max = min(frame_height, int(max(hand_y_coords) + hand_padding))
        
        # Get wrist and fingertip positions for direction
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # Calculate hand direction vector from wrist to middle of fingers
        center_x = (index_tip.x + pinky_tip.x) / 2
        center_y = (index_tip.y + pinky_tip.y) / 2
        direction_x = center_x - wrist.x
        direction_y = center_y - wrist.y
        
        # Extend search area in the direction of the fingers
        extension_factor = 2.0
        search_x = wrist.x + direction_x * extension_factor
        search_y = wrist.y + direction_y * extension_factor
        
        # Get larger ROI for item detection
        roi_x_min = max(0, min(hand_x_min, int(search_x * frame_width - padding)))
        roi_y_min = max(0, min(hand_y_min, int(search_y * frame_height - padding)))
        roi_x_max = min(frame_width, max(hand_x_max, int(search_x * frame_width + padding)))
        roi_y_max = min(frame_height, max(hand_y_max, int(search_y * frame_height + padding)))
        
        # Extract ROI
        roi = frame[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        if roi.size == 0:
            return frame
        
        # Convert to grayscale
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Create hand mask in ROI coordinates
        hand_mask = np.zeros_like(gray_roi)
        hand_points = np.array([[int(x * frame_width) - roi_x_min, int(y * frame_height) - roi_y_min] 
                              for x, y in zip(hand_x_coords, hand_y_coords)], dtype=np.int32)
        cv2.fillConvexPoly(hand_mask, hand_points, 255)
        
        # Dilate hand mask
        hand_kernel = np.ones((5,5), np.uint8)
        dilated_hand = cv2.dilate(hand_mask, hand_kernel, iterations=2)
        
        # Create item search mask (exclude hand area)
        item_search_mask = cv2.bitwise_not(dilated_hand)
        
        # Apply corner detection only in the item search area
        corners = cv2.goodFeaturesToTrack(
            gray_roi,
            mask=item_search_mask,
            maxCorners=50,
            qualityLevel=0.04,
            minDistance=10,
            blockSize=3,
            useHarrisDetector=True,
            k=0.04
        )
        
        # Create corner mask
        corner_mask = np.zeros_like(gray_roi)
        if corners is not None:
            corners = np.int0(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(corner_mask, (x, y), 5, 255, -1)
        
        # Apply edge detection
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 150)
        
        # Combine all features
        kernel = np.ones((5,5), np.uint8)
        dilated_corners = cv2.dilate(corner_mask, kernel, iterations=2)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Create combined mask ensuring hand is included
        combined_mask = cv2.bitwise_or(cv2.bitwise_or(dilated_hand, dilated_edges), dilated_corners)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours found, return the hand region
            return frame[hand_y_min:hand_y_max, hand_x_min:hand_x_max]
        
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get convex hull
        hull = cv2.convexHull(largest_contour)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(hull)
        
        # Ensure the hand region is included in the final crop
        x = min(x, hand_x_min - roi_x_min)
        y = min(y, hand_y_min - roi_y_min)
        w = max(w, (hand_x_max - hand_x_min) + 2 * hand_padding)
        h = max(h, (hand_y_max - hand_y_min) + 2 * hand_padding)
        
        # Add padding to the final crop
        crop_padding = 20
        x = max(0, x - crop_padding)
        y = max(0, y - crop_padding)
        w = min(roi.shape[1] - x, w + 2 * crop_padding)
        h = min(roi.shape[0] - y, h + 2 * crop_padding)
        
        # Extract final crop
        final_crop = roi[y:y+h, x:x+w]
        
        # Debug visualization
        # debug_frame = roi.copy()
        # cv2.drawContours(debug_frame, [hand_points], -1, (255, 0, 0), 2)  # Hand in blue
        # if corners is not None:
        #     for corner in corners:
        #         x, y = corner.ravel()
        #         cv2.circle(debug_frame, (x, y), 3, (0, 255, 0), -1)  # Corners in green
        # cv2.drawContours(debug_frame, [hull], 0, (0, 0, 255), 2)  # Hull in red
        # cv2.imshow('Detection Steps', debug_frame)
        # cv2.imshow('Hand Mask', dilated_hand)
        # cv2.imshow('Item Search Area', item_search_mask)
        # cv2.imshow('Corner Detection', corner_mask)
        # cv2.imshow('Combined Mask', combined_mask)
        
        return final_crop

    def save_current_frames(self, frame1, frame2, raw_frame1, raw_frame2):
        """Save both raw and interface frames if recording"""
        if self.is_recording and self.current_sequence_dir:
            # Use frame count as index for this sequence
            frame_index = f"frame_{self.frame_count:04d}"
            
            # Determine which camera is palm and which is back
            if self.palm_camera == "External CAM":
                palm_frame = frame1
                back_frame = frame2
                palm_raw = raw_frame1
                back_raw = raw_frame2
                palm_landmarks = self.cam1_state.hand_landmarks
                back_landmarks = self.cam2_state.hand_landmarks
            else:
                palm_frame = frame2
                back_frame = frame1
                palm_raw = raw_frame2
                back_raw = raw_frame1
                palm_landmarks = self.cam2_state.hand_landmarks
                back_landmarks = self.cam1_state.hand_landmarks
            
            # Save interface frames
            cv2.imwrite(os.path.join(self.current_sequence_dir, "interface", "palm", f"{frame_index}.jpg"), palm_frame)
            cv2.imwrite(os.path.join(self.current_sequence_dir, "interface", "back", f"{frame_index}.jpg"), back_frame)
            
            # Save raw frames
            cv2.imwrite(os.path.join(self.current_sequence_dir, "raw", "palm", f"{frame_index}.jpg"), palm_raw)
            cv2.imwrite(os.path.join(self.current_sequence_dir, "raw", "back", f"{frame_index}.jpg"), back_raw)
            
            # Save cropped palm view
            if palm_landmarks:
                cropped_palm = self.get_hand_crop(palm_raw, palm_landmarks)
                cv2.imwrite(os.path.join(self.current_sequence_dir, "cropped", "palm", f"{frame_index}.jpg"), cropped_palm)
            
            # Save cropped back view
            if back_landmarks:
                cropped_back = self.get_hand_crop(back_raw, back_landmarks)
                cv2.imwrite(os.path.join(self.current_sequence_dir, "cropped", "back", f"{frame_index}.jpg"), cropped_back)
            
            self.frame_count += 1

    def stop_recording(self):
        """Stop recording sequence"""
        self.is_recording = False
        if self.current_sequence_dir and os.path.exists(self.current_sequence_dir):
            # Discard sequences with less than 10 frames
            if self.frame_count < 10:
                print(f"Discarding sequence {self.sequence_time} - too few frames ({self.frame_count})")
                shutil.rmtree(self.current_sequence_dir)
            else:
                # Move sequence to appropriate directory based on action
                if self.action_zone == "taking_out":
                    target_dir = self.taking_dir
                    action_type = "TAKING OUT"
                else:
                    target_dir = self.putting_dir
                    action_type = "PUTTING IN"
                
                # Move the sequence to the correct directory
                target_path = os.path.join(target_dir, self.sequence_time)
                shutil.move(self.current_sequence_dir, target_path)
                
                print(f"Completed recording {action_type} sequence {self.sequence_time} with {self.frame_count} frames")
            
            # Clean up
            self.current_sequence_dir = None
            self.frame_count = 0

    def filter_hand_detections(self, yolo_results):
        """Filter YOLO results to only include hand detections with high confidence"""
        if yolo_results.pred[0] is None:
            return None
        
        # Filter for hand class (assuming hand is class 0) with confidence > 0.5
        hand_detections = yolo_results.pred[0][yolo_results.pred[0][:, 5] == 0]
        high_conf_hands = hand_detections[hand_detections[:, 4] > 0.5]
        
        return high_conf_hands if len(high_conf_hands) > 0 else None

    def normalize_mediapipe_confidence(self, confidence):
        """Normalize MediaPipe confidence scores to be closer to 1.0"""
        # Adjust these parameters to get desired confidence distribution
        min_conf = 0.5
        max_conf = 1.0
        
        return np.clip((confidence - min_conf) / (max_conf - min_conf), 0, 1)

    def plot_performance_graphs(self):
        """Generate performance comparison graphs"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Processing Time Distribution
        sns.kdeplot(data=list(self.mp_times), ax=ax1, color='green', 
                   label=f'MediaPipe (avg: {np.mean(self.mp_times):.1f}ms)')
        sns.kdeplot(data=list(self.yolo_times), ax=ax1, color='blue', 
                   label=f'YOLO (avg: {np.mean(self.yolo_times):.1f}ms)')
        ax1.set_title('Processing Time Distribution')
        ax1.set_xlabel('Time (milliseconds)')
        ax1.set_ylabel('Density')
        ax1.legend()
        
        # Confidence Score Distribution
        sns.kdeplot(data=list(self.mp_confidences), ax=ax2, color='green', 
                   label=f'MediaPipe (avg: {np.mean(self.mp_confidences):.2f})')
        sns.kdeplot(data=list(self.yolo_confidences), ax=ax2, color='blue', 
                   label=f'YOLO (avg: {np.mean(self.yolo_confidences):.2f})')
        ax2.set_title('Confidence Score Distribution (1.0 = High Confidence)')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # Landmark Confidence Heatmap
        landmark_names = [f'LANDMARK_{i}' for i in range(21)]
        landmark_data = np.hstack((
            self.landmark_confidences['MediaPipe'] / self.frame_count,
            self.landmark_confidences['YOLO'] / self.frame_count
        ))
        
        sns.heatmap(landmark_data, ax=ax3, cmap='RdYlGn', 
                   xticklabels=['MediaPipe', 'YOLO'],
                   yticklabels=landmark_names,
                   annot=True, fmt='.3f')
        ax3.set_title('Per-Landmark Detection Confidence')
        
        plt.tight_layout()
        plt.savefig('performance_comparison.png')
        plt.close()

    def run(self):
        print("\nStarting dual camera hand detection")
        print("Press 'q' to quit")
        print(f"Saving captured frames to: {self.base_dir}")
        
        # Start camera threads
        self.cam1_thread.start()
        self.cam2_thread.start()
        
        # FPS calculation variables
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        while True:
            # Get synchronized frames from both cameras
            try:
                ret1, raw_frame1 = self.frame_queue1.get(timeout=0.1)
                ret2, raw_frame2 = self.frame_queue2.get(timeout=0.1)
                
                # Update FPS calculation
                frame_count += 1
                if frame_count % 30 == 0:  # Calculate FPS every 30 frames
                    elapsed = time.time() - fps_start_time
                    fps = frame_count / elapsed
                    print(f"\rSystem FPS: {fps:.1f}", end="")
                    frame_count = 0
                    fps_start_time = time.time()
                
            except:
                continue
                
            if not ret1 or not ret2:
                continue
                
            # Process frames with separate hand detectors
            # External camera (left) - no mirroring
            frame1 = self.process_frame(raw_frame1.copy(), self.cam1_state, self.trail_points_cam1, 
                                      "External CAM", self.hands1, is_mirrored=False)
            
            # Built-in camera (right) - mirror for display and detection
            frame2 = self.process_frame(raw_frame2.copy(), self.cam2_state, self.trail_points_cam2, 
                                      "Built-in CAM", self.hands2, is_mirrored=True)
            
            # Save frames if recording
            if self.is_recording:
                self.save_current_frames(frame1, frame2, raw_frame1, raw_frame2)
            
            # Check for action
            action = self.check_action()
            if action:
                self.current_action = action
            else:
                self.current_action = None
            
            # Ensure both frames have the same size for display
            target_height = 480
            target_width = int(target_height * (frame1.shape[1] / frame1.shape[0]))
            
            frame1 = cv2.resize(frame1, (target_width, target_height))
            frame2 = cv2.resize(frame2, (target_width, target_height))
            
            # Create combined display
            combined_frame = np.hstack((frame1, frame2))
            
            # Add separator line
            mid_x = combined_frame.shape[1] // 2
            cv2.line(combined_frame, (mid_x, 0), (mid_x, combined_frame.shape[0]), 
                    (100, 100, 100), 2)
            
            # Add recording and action indicators
            if self.is_recording:
                status_text = "RECORDING"
                cv2.putText(combined_frame, status_text, 
                          (mid_x - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1.0, (0, 0, 255), 2)
            elif self.current_action:
                action_text = "TAKING OUT" if self.current_action == "taking_out" else "PUTTING IN"
                cv2.putText(combined_frame, action_text, 
                          (mid_x - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1.0, (0, 255, 0), 2)
            
            # Show combined frame
            cv2.imshow('Dual Camera Hand Detection', combined_frame)
            
            self.frame_count += 1
            
            # Generate performance graphs every 1000 frames
            if self.frame_count % 1000 == 0:
                self.plot_performance_graphs()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # Generate final performance graphs before exit
                self.plot_performance_graphs()
                break
        
        # Cleanup
        self.cam1_thread.stopped = True
        self.cam2_thread.stopped = True
        self.cam1_thread.join()
        self.cam2_thread.join()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    detector = DualCameraHandDetector()
    detector.run() 