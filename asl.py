import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import math

class ASLDetector:
    def __init__(self):
        # Initialize MediaPipe hands module
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Store hand landmarks for gesture recognition
        self.landmark_history = deque(maxlen=10)
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        radians = math.atan2(c[1] - b[1], c[0] - b[0]) - \
                  math.atan2(a[1] - b[1], a[0] - b[0])
        angle = math.degrees(radians)
        angle = abs(angle)
        if angle > 180:
            angle = 360 - angle
        return angle
    
    def detect_asl_letter(self, hand_landmarks):
        if not hand_landmarks:
            return None
        
        # Get landmark positions
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.append([lm.x, lm.y, lm.z])
        
        # Define landmark indices
        WRIST = 0
        THUMB_CMC = 1
        THUMB_MCP = 2
        THUMB_IP = 3
        THUMB_TIP = 4
        INDEX_MCP = 5
        INDEX_PIP = 6
        INDEX_DIP = 7
        INDEX_TIP = 8
        MIDDLE_MCP = 9
        MIDDLE_PIP = 10
        MIDDLE_DIP = 11
        MIDDLE_TIP = 12
        RING_MCP = 13
        RING_PIP = 14
        RING_DIP = 15
        RING_TIP = 16
        PINKY_MCP = 17
        PINKY_PIP = 18
        PINKY_DIP = 19
        PINKY_TIP = 20
        
        # Helper function to check if finger is extended
        def is_finger_up(tip_idx, pip_idx):
            return landmarks[tip_idx][1] < landmarks[pip_idx][1]
        
        # Helper function to check if thumb is extended (special case)
        def is_thumb_up():
            # Check both x and y coordinates for thumb
            thumb_extended = landmarks[THUMB_TIP][0] > landmarks[THUMB_IP][0] + 0.02
            return thumb_extended
        
        # Helper function to calculate distance between two points
        def get_distance(p1_idx, p2_idx):
            return np.sqrt(
                (landmarks[p1_idx][0] - landmarks[p2_idx][0])**2 +
                (landmarks[p1_idx][1] - landmarks[p2_idx][1])**2 +
                (landmarks[p1_idx][2] - landmarks[p2_idx][2])**2
            )
        
        # Helper function to check if fingers are bent
        def is_finger_bent(tip_idx, dip_idx, pip_idx):
            angle = self.calculate_angle(
                landmarks[tip_idx][:2], 
                landmarks[dip_idx][:2], 
                landmarks[pip_idx][:2]
            )
            return angle < 160  # Finger is bent if angle is less than 160 degrees
        
        # Check which fingers are up
        fingers_up = []
        fingers_up.append(1 if is_thumb_up() else 0)
        fingers_up.append(1 if is_finger_up(INDEX_TIP, INDEX_PIP) else 0)
        fingers_up.append(1 if is_finger_up(MIDDLE_TIP, MIDDLE_PIP) else 0)
        fingers_up.append(1 if is_finger_up(RING_TIP, RING_PIP) else 0)
        fingers_up.append(1 if is_finger_up(PINKY_TIP, PINKY_PIP) else 0)
        
        # Additional measurements for complex letters
        thumb_index_distance = get_distance(THUMB_TIP, INDEX_TIP)
        thumb_middle_distance = get_distance(THUMB_TIP, MIDDLE_TIP)
        index_middle_distance = get_distance(INDEX_TIP, MIDDLE_TIP)
        thumb_pinky_distance = get_distance(THUMB_TIP, PINKY_TIP)
        
        # Check for specific hand orientations
        palm_facing_forward = landmarks[MIDDLE_MCP][2] < landmarks[WRIST][2]
        
        # Detect each letter A-Z
        
        # A - Fist with thumb on side
        if fingers_up == [1, 0, 0, 0, 0] and thumb_index_distance < 0.08:
            return "A"
        
        # B - Flat hand with thumb across palm
        elif fingers_up == [0, 1, 1, 1, 1] and landmarks[THUMB_TIP][0] < landmarks[INDEX_MCP][0]:
            return "B"
        
        # C - Curved hand shape
        elif all(is_finger_bent(tip, dip, pip) for tip, dip, pip in [
            (INDEX_TIP, INDEX_DIP, INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP),
            (RING_TIP, RING_DIP, RING_PIP),
            (PINKY_TIP, PINKY_DIP, PINKY_PIP)
        ]) and thumb_index_distance > 0.05:
            return "C"
        
        # D - Index up, others closed, thumb touches middle finger
        elif fingers_up == [0, 1, 0, 0, 0] and thumb_middle_distance < 0.05:
            return "D"
        
        # E - All fingers bent down, thumb across fingers
        elif all(f == 0 for f in fingers_up) and landmarks[THUMB_TIP][1] > landmarks[INDEX_PIP][1]:
            return "E"
        
        # F - Index and middle touch thumb, others up
        elif (fingers_up[3] == 1 and fingers_up[4] == 1 and 
            thumb_index_distance < 0.05 and fingers_up[1] == 0):
            return "F"
        
        # G - Index pointing sideways, thumb parallel
        elif (fingers_up[1] == 1 and fingers_up[0] == 1 and 
            all(f == 0 for f in fingers_up[2:]) and 
            abs(landmarks[INDEX_TIP][1] - landmarks[THUMB_TIP][1]) < 0.05):
            return "G"
        
        # H - Index and middle horizontal, pointing sideways
        elif (fingers_up[1] == 1 and fingers_up[2] == 1 and 
            all(f == 0 for f in fingers_up[3:]) and fingers_up[0] == 0 and
            abs(landmarks[INDEX_TIP][1] - landmarks[MIDDLE_TIP][1]) < 0.03):
            return "H"
        
        # I - Pinky up only, thumb across
        elif fingers_up == [0, 0, 0, 0, 1] and landmarks[THUMB_TIP][0] < landmarks[MIDDLE_MCP][0]:
            return "I"
        
        # J - Pinky up with downward motion (requires motion tracking)
        # This would need motion history to detect properly
        elif fingers_up == [0, 0, 0, 0, 1]:
            return "J (static)"
        
        # K - Index and middle up in V, thumb touches middle
        elif (fingers_up[1] == 1 and fingers_up[2] == 1 and 
            all(f == 0 for f in fingers_up[3:]) and 
            get_distance(THUMB_TIP, MIDDLE_PIP) < 0.05):
            return "K"
        
        # L - L shape with thumb and index
        elif fingers_up == [1, 1, 0, 0, 0]:
            angle = self.calculate_angle(
                landmarks[THUMB_TIP][:2], 
                landmarks[INDEX_MCP][:2], 
                landmarks[INDEX_TIP][:2]
            )
            if 70 < angle < 110:
                return "L"
        
        # M - Three fingers over thumb
        elif (all(f == 0 for f in fingers_up[1:4]) and 
            landmarks[THUMB_TIP][1] > landmarks[RING_DIP][1]):
            return "M"
        
        # N - Two fingers over thumb
        elif (all(f == 0 for f in fingers_up[1:3]) and fingers_up[3] == 0 and
            landmarks[THUMB_TIP][1] > landmarks[MIDDLE_DIP][1]):
            return "N"
        
        # O - All fingers and thumb form circle
        elif (thumb_index_distance < 0.05 and 
            all(is_finger_bent(tip, dip, pip) for tip, dip, pip in [
                (INDEX_TIP, INDEX_DIP, INDEX_PIP),
                (MIDDLE_TIP, MIDDLE_DIP, MIDDLE_PIP),
                (RING_TIP, RING_DIP, RING_PIP),
                (PINKY_TIP, PINKY_DIP, PINKY_PIP)
            ])):
            return "O"
        
        # P - Similar to K but pointing down
        elif (fingers_up[1] == 1 and fingers_up[2] == 1 and 
            landmarks[INDEX_TIP][1] > landmarks[WRIST][1] and
            get_distance(THUMB_TIP, MIDDLE_PIP) < 0.05):
            return "P"
        
        # Q - Similar to G but pointing down
        elif (fingers_up[1] == 1 and fingers_up[0] == 1 and
            landmarks[INDEX_TIP][1] > landmarks[WRIST][1]):
            return "Q"
        
        # R - Index and middle crossed
        elif (fingers_up[1] == 1 and fingers_up[2] == 1 and
            abs(landmarks[INDEX_TIP][0] - landmarks[MIDDLE_TIP][0]) < 0.02):
            return "R"
        
        # S - Fist with thumb over fingers
        elif (all(f == 0 for f in fingers_up) and 
            landmarks[THUMB_TIP][1] < landmarks[INDEX_PIP][1]):
            return "S"
        
        # T - Thumb between index and middle
        elif (all(f == 0 for f in fingers_up[1:]) and
            landmarks[THUMB_TIP][0] > landmarks[INDEX_MCP][0] and
            landmarks[THUMB_TIP][0] < landmarks[MIDDLE_MCP][0]):
            return "T"
        
        # U - Index and middle up together
        elif (fingers_up[1] == 1 and fingers_up[2] == 1 and
            index_middle_distance < 0.03 and
            all(f == 0 for f in fingers_up[3:])):
            return "U"
        
        # V - Index and middle up in V shape
        elif (fingers_up[1] == 1 and fingers_up[2] == 1 and
            index_middle_distance > 0.05 and
            all(f == 0 for f in fingers_up[3:])):
            return "V"
        
        # W - Three fingers up
        elif fingers_up == [0, 1, 1, 1, 0]:
            return "W"
        
        # X - Index bent hook shape
        elif (is_finger_bent(INDEX_TIP, INDEX_DIP, INDEX_PIP) and
            all(f == 0 for f in fingers_up[2:])):
            return "X"
        
        # Y - Thumb and pinky extended
        elif fingers_up == [1, 0, 0, 0, 1] and thumb_pinky_distance > 0.1:
            return "Y"
        
        # Z - Index traces Z pattern (requires motion)
        # This would need motion history to detect properly
        elif fingers_up == [0, 1, 0, 0, 0]:
            return "Z (static)"
        
        # Default
        return None
    
    def process_frame(self, frame):
        """Process a single frame and detect hand gestures"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Convert back to BGR for OpenCV
        rgb_frame.flags.writeable = True
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        
        detected_letters = []
        
        # Draw hand landmarks and detect gestures
        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Detect ASL letter
                letter = self.detect_asl_letter(hand_landmarks)
                if letter:
                    detected_letters.append(letter)
                    
                    # Get hand position for text display
                    h, w, _ = frame.shape
                    cx = int(hand_landmarks.landmark[9].x * w)  # Middle of hand
                    cy = int(hand_landmarks.landmark[9].y * h)
                    
                    # Display detected letter
                    cv2.putText(frame, letter, (cx - 20, cy - 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
        
        return frame, detected_letters

def main():
    # Initialize ASL detector
    asl_detector = ASLDetector()
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Word builder for continuous recognition
    current_word = ""
    last_letter = None
    letter_count = 0
    
    print("ASL Detection Started. Press 'q' to quit.")
    print("Press 'space' to add current letter to word, 'c' to clear word")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Process frame
        processed_frame, detected_letters = asl_detector.process_frame(frame)
        
        # Handle letter detection for word building
        if detected_letters:
            current_letter = detected_letters[0]
            if current_letter == last_letter:
                letter_count += 1
            else:
                letter_count = 1
                last_letter = current_letter
        
        # Display current word being built
        cv2.putText(processed_frame, f"Word: {current_word}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display instructions
        cv2.putText(processed_frame, "Space: Add letter | C: Clear | Q: Quit", 
                   (10, processed_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('ASL Detection', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' ') and last_letter and letter_count > 5:
            # Add letter to word if held for sufficient frames
            current_word += last_letter
            print(f"Added '{last_letter}' to word: {current_word}")
        elif key == ord('c'):
            current_word = ""
            print("Word cleared")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Install required packages if not already installed
    # pip install opencv-python mediapipe numpy
    
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have installed the required packages:")
        print("pip install opencv-python mediapipe numpy")