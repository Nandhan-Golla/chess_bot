import cv2
import numpy as np
import depthai as dai

# Chessboard dimensions
BOARD_SIZE = (8, 8)  # 8x8 inner corners

def setup_pipeline():
    pipeline = dai.Pipeline()
    
    mono = pipeline.create(dai.node.MonoCamera)
    mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    
    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("mono")
    mono.out.link(xout.input)
    
    return pipeline

def preprocess_frame(frame):
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    
    # Enhance contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    gray = cv2.equalizeHist(gray)
    
    # Adaptive thresholding with better parameters
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 5
    )
    
    return gray, thresh

def detect_chessboard_corners(frame):
    gray, thresh = preprocess_frame(frame)
    
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FAST_CHECK
    )
    
    ret, corners = cv2.findChessboardCorners(thresh, BOARD_SIZE, flags=flags)
    
    if ret:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
    
    return ret, corners, gray, thresh

def label_squares(frame, corners):
    if corners is None:
        return frame
    
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    corners = corners.reshape(BOARD_SIZE[1], BOARD_SIZE[0], 2)
    files = 'abcdefgh'
    ranks = '12345678'
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (0, 255, 0)
    thickness = 1
    
    for i in range(BOARD_SIZE[1]):
        for j in range(BOARD_SIZE[0]):
            top_left = corners[i, j]
            bottom_right = corners[min(i + 1, BOARD_SIZE[1] - 1), min(j + 1, BOARD_SIZE[0] - 1)]
            center_x = int((top_left[0] + bottom_right[0]) / 2)
            center_y = int((top_left[1] + bottom_right[1]) / 2)
            square_label = f"{files[j]}{ranks[7-i]}"
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
    pipeline = setup_pipeline()
    
    with dai.Device(pipeline) as device:
        mono_queue = device.getOutputQueue(name="mono", maxSize=4, blocking=False)
        
        while True:
            in_mono = mono_queue.tryGet()
            if in_mono is not None:
                frame = in_mono.getCvFrame()
                
                ret, corners, gray, thresh = detect_chessboard_corners(frame)
                
                display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                
                if ret:
                    cv2.drawChessboardCorners(display_frame, BOARD_SIZE, corners, ret)
                    display_frame = label_squares(display_frame, corners)
                    status = "Chessboard Detected"
                    status_color = (0, 255, 0)
                else:
                    status = "Chessboard Not Detected"
                    status_color = (0, 0, 255)
                
                cv2.putText(
                    display_frame,
                    status,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_color,
                    2
                )
                
                # Show thresholded image for debugging
                cv2.imshow("Thresholded", thresh)
                cv2.imshow("Chessboard Detection", display_frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()