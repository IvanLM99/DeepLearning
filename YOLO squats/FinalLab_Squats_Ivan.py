# Versión completamente funcional en local del contador de sentadillas
# Se ejecuta sobre entorno virtual con python 3.10 y ultralytics 8.0.201

##################################################
# Libraries
##################################################

import math
from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

##################################################
# Global variables and general set up
##################################################

# Body parts ordered as indicated in keypoints
idx2bparts = ["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
              "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
              "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", "Left Knee",
              "Right Knee", "Left Ankle", "Right Ankle"]

# Index of body parts
bparts2idx = {key: ix for ix, key in enumerate(idx2bparts)}

# State and squat count
STATE = 'UP'
COUNT = 0
state_stack = deque(maxlen=6)
CHECK = True  # Used for debugging
ONE_IMAGE = False
BAD_SQUAT_FLAG = False  # New flag to track bad squat state

# Load the Yolov8 model
model = YOLO('yolov8s-pose.pt')

# Open the video file or webcam
#source = 0  # Webcam input
source = "IvanSquat.mp4"  # Video file input

# Squat counting modes
MODE = 2

##################################################
# Helper functions
##################################################

def add_annotations(frame, bad_squat=False):
    """
    Add state (up/down) and squats count (number) to the image.

    Args:
        frame (numpy array): Current frame captured

    Returns:
        frame with added text
    """
    state_text = f"State: {STATE}"
    count_text = f"Count: {COUNT}"
    mode_text = f"Mode: {MODE}"
    bad_squat_text = "Bad Squatting" if bad_squat else ""

    text_position1 = (10, 30)
    text_position2 = (10, 60)
    text_position3 = (10, 90)
    text_position4 = (10, 120)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    green = (0, 255, 0)
    red = (0, 0, 255)
    font_color = green if STATE == 'UP' else red
    font_thickness = 2

    frame_with_text = frame.copy()
    cv2.putText(frame_with_text, state_text, text_position1, font,
                font_scale, font_color, font_thickness)
    cv2.putText(frame_with_text, count_text, text_position2, font,
                font_scale, green, font_thickness)
    cv2.putText(frame_with_text, mode_text, text_position3, font,
                font_scale, green, font_thickness)
    cv2.putText(frame_with_text, bad_squat_text, text_position4, font,
                font_scale, red, font_thickness)

    return frame_with_text

def legs_angles(left, right, verbose=False):
    """
    It calculates the minimum angle that make up the vector hip-knee with
    the vector knee-ankle in each leg. The inputs are numpy arrays with
    shape 3x2 (3 points x 2 coordinates) and the output is a numpy array
    of shape [2,] with each angle in degrees.

    Args:
        left (numpy array): Coordinates of joints hip, knee and ankle of
            the left leg. The matrix has the following shape:
            [x hip  , y hip  ]
            [x knee , y knee ]
            [x ankle, y ankle]
        right (numpy array): Coordinates of joints hip, knee and ankle of
            the right leg. The matrix has the same shape as 'left'
        verbose (bool, optional): Print info. Defaults to False.

    Returns:
        A numpy array with shape [2,] with the angles of the two legs in
            degrees.
    """
    angles = []
    for v in [left, right]:
        x1, y1 = v[0, 0], v[0, 1]
        x2, y2 = v[1, 0], v[1, 1]
        x3, y3 = v[2, 0], v[2, 1]

        vector1 = (x1 - x2, y1 - y2)
        vector2 = (x3 - x2, y3 - y2)

        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        magnitude1 = math.sqrt(vector1[0]**2 + vector1[1]**2)
        magnitude2 = math.sqrt(vector2[0]**2 + vector2[1]**2)

        cosine_theta = dot_product / (magnitude1 * magnitude2)
        theta_radians = math.acos(max(-1, min(cosine_theta, 1)))
        theta_degrees = math.degrees(theta_radians)

        angles.append(theta_degrees)

        if verbose:
            print(f"The angle in the knee (triangle knee-hip-ankle) is {theta_degrees:.2f} degrees.")

    return np.array(angles)

def get_legs_coords(kpts):
    """
    It gets the keypoints of the result object and extract those from
    hip, knee and ankle of left and right legs. The outputs are np arrays
    with the coordinates x, y and the confidence value

    Args:
        kpts (ultralytics keypoints): Keypoints object from the Result
            object in a pose estimation.

    Returns:
        left_leg_coords (numpy array): 3x3 numpy array with the coordinates
            (x, y, confidence) of the left hip, left knee and left ankle
            in the image
        left_leg_coords (numpy array): 3x3 numpy array with the coordinates
            (x, y, confidence) of the left hip, left knee and left ankle
            in the image
    """
    # Indices of left and right hip, knee and ankle
    left_leg = [11, 13, 15]
    right_leg = [12, 14, 16]
    # Left leg
    left_leg_coords = kpts.data[0, left_leg, :].cpu().numpy()
    # Right leg
    right_leg_coords = kpts.data[0, right_leg, :].cpu().numpy()
    return left_leg_coords, right_leg_coords

def get_hand_positions(kpts):
    """
    Get the coordinates of the wrists and shoulders.
    """
    left_wrist = kpts.data[0, bparts2idx['Left Wrist'], :2].cpu().numpy()
    right_wrist = kpts.data[0, bparts2idx['Right Wrist'], :2].cpu().numpy()
    left_shoulder = kpts.data[0, bparts2idx['Left Shoulder'], :2].cpu().numpy()
    right_shoulder = kpts.data[0, bparts2idx['Right Shoulder'], :2].cpu().numpy()
    left_hip = kpts.data[0, bparts2idx['Left Hip'], :2].cpu().numpy()
    right_hip = kpts.data[0, bparts2idx['Right Hip'], :2].cpu().numpy()

    return left_wrist, right_wrist, left_shoulder, right_shoulder, left_hip, right_hip

def evaluate_position(result, limit_conf=0.3, verbose=False):
    """
    Evaluate position of the body in the image

    It updates the global variables STATE (UP or DOWN) and the number
    of squats done (COUNT)

    Args:
        result (Ultralytics Results): Results object from Ultralytics. It
            contains all the data of the pose estimation.
        limit_conf (float, optional): It's the limiting confidence. Greater
            confidences in (all) points estimation will be considered,
            otherwise they will be descarted. Defaults to 0.3.
        verbose (bool, optional): Print info. Defaults to False.
    """

    # Global variables
    global COUNT, STATE, state_stack, BAD_SQUAT_FLAG
    bad_squat = False

    # Loop through Ultralytics Results
    for r in result:
        # Get bounding boxes
        box = r.boxes
        if r.names[int(box.cls.item())] != 'person':
            break
        # Get keypoints
        kpts = r.keypoints # Keypoints object for pose output
        # Get coordinates of the joints of the left and right legs
        left_coords, right_coords = get_legs_coords(kpts)

        # Check for confidences
        if (left_coords[:, 2] > limit_conf).all() and (right_coords[:, 2] > limit_conf).all():

            # Calculate the minimum angle in both legs
            angles = legs_angles(left_coords[:, :2], right_coords[:, :2])
            # Calculate positions of wrists, shoulders and hip
            left_wrist, right_wrist, left_shoulder, right_shoulder, left_hip, right_hip = get_hand_positions(kpts)

            # Legs bent or stretched
            if (angles < 120).all() and STATE == 'UP':
                STATE = 'DOWN'
            elif (angles > 160).all() and STATE == 'DOWN':
                STATE = 'UP'

            if MODE == 2:
                if (left_wrist[1] > left_hip[1] or right_wrist[1] > right_hip[1]):
                    bad_squat = True
                    
            elif MODE == 3:
                if (left_wrist[1] > left_shoulder[1] or right_wrist[1] > right_shoulder[1]):
                    bad_squat = True
            
            if bad_squat:
                BAD_SQUAT_FLAG = True
            if state_stack == deque(['UP', 'UP', 'UP', 'UP', 'UP', 'UP']):
                BAD_SQUAT_FLAG = False
                
            # Update stack of states and count
            state_stack.append(STATE)
            if len(state_stack) == 6:
                if state_stack == deque(['DOWN', 'UP', 'UP', 'UP', 'UP', 'UP']):
                    if not BAD_SQUAT_FLAG:
                        COUNT += 1
                    BAD_SQUAT_FLAG = False  # Reset flag after a valid squat cycle

    if verbose:
        print(f"State: {STATE}")
        print(f"Count: {COUNT}")

    return bad_squat

##################################################
# Main program
##################################################

# Select source
cap = cv2.VideoCapture(source)
stream = True  # If stream=True the output is a generator
               # otherwise it's a list

# Loop through the video frames
cont = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    # If the frame is empty, break the loop
    if not success:
        break
    
    # Perform pose estimation on this single frame
    results = model(source=frame, show=True, conf=0.3, save=False, stream=stream)

    # Extract data from results
    if not stream:
        r = results[0]
    else:
        r = next(results)

    if cont == 0:
        if ONE_IMAGE:
            cv2.destroyWindow('image0.jpg')
        else:
            cv2.setWindowTitle('image0.jpg', 'YoloV8 Results')

    # Evaluate position
    bad_squat = evaluate_position(r)

    frame_with_text = add_annotations(frame, bad_squat)

    # Display the annotated frame
    cv2.imshow('Squat Counter Window', frame_with_text)

    # Check for user input to break the loop (e.g., press 'q' to exit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Increment frame counter
    cont += 1

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
