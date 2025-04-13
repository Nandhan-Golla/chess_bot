import cv2
import numpy as np
import depthai as dai
from pathlib import Path

# Chessboard dimensions
BOARD_SIZE = (8, 8)  # Standard 8x8 chessboard
SQUARE_SIZE_MM = 28.5 / 8 * 10  # 28.5cm board, 8 squares, convert to mm

def setup_pipeline():
    pipeline = dai.Pipeline()
    
    # Define mono camera
    mono = pipeline.create(dai.node.MonoCamera)
    mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    
    # Create output
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("mono")
    mono.out.link(xout.input)
    
    return pipeline

def detect_chessboard_corners(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gaussian_blur,
        BOARD_SIZE,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        # Refine corners
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
    return ret, corners

def label_squares(frame, corners):
    if corners is None:
        return frame
    
    # Reshape corners to 8x8 grid
    corners = corners.reshape(BOARD_SIZE[1], BOARD_SIZE[0], 2)
    
    # Define chessboard coordinates (a1 to h8)
    files = 'abcdefgh'
    ranks = '12345678'
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 255, 0)  # Green labels
    thickness = 1
    
    # Label each square (center of the square)
    for i in range(BOARD_SIZE[1]):  # rows (ranks)
        for j in range(BOARD_SIZE[0]):  # cols (files)
            # Get square corners
            top_left = corners[i, j]
            bottom_right = corners[min(i + 1, BOARD_SIZE[1] - 1), min(j + 1, BOARD_SIZE[0] - 1)]
            
            # Calculate center of the square
            center_x = int((top_left[0] + bottom_right[0]) / 2)
            center_y = int((top_left[1] + bottom_right[1]) / 2)
            
            # Square label (e.g., e4)
            square_label = f"{files[j]}{ranks[7-i]}"  # Adjust for chess notation
            
            # Put text on frame
            cv2.putText(
                frame,
                square_label,
                (center_x - 10, center_y + 5),
                font,
                font_scale,
                color,
                thickness,
                cv2.LINE_AA
            )
    
    return frame

def main():
    # Setup DepthAI pipeline
    pipeline = setup_pipeline()
    
    # Connect to device
    with dai.Device(pipeline) as device:
        # Get output queue
        mono_queue = device.getOutputQueue(name="mono", maxSize=4, blocking=False)
        
        while True:
            # Get frame
            in_mono = mono_queue.tryGet()
            if in_mono is not None:
                # Convert to OpenCV format
                frame = in_mono.getCvFrame()
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                
                # Detect chessboard
                ret, corners = detect_chessboard_corners(frame)
                
                if ret:
                    # Draw corners
                    cv2.drawChessboardCorners(frame, BOARD_SIZE, corners, ret)
                    
                    # Label squares
                    frame = label_squares(frame, corners)
                
                # Display frame
                cv2.imshow("Chessboard Detection", frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()