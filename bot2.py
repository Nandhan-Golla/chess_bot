import cv2
import depthai as dai
import numpy as np
from time import sleep

# Initialize Oak-D Lite with best focus
def init_oakd(focus_value):
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    cam_rgb.setPreviewSize(1024, 1024)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(30)
    cam_rgb.initialControl.setManualFocus(focus_value)
    cam_rgb.preview.link(xout_rgb.input)
    return pipeline

# Detect chessboard with enhanced robustness
def detect_chessboard(frame, debug_idx):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Save raw frame and check sharpness
    cv2.imwrite(f"debug_raw_{debug_idx}.jpg", frame)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Frame sharpness: {sharpness:.2f} (target >100)")
    
    # Enhanced edge detection
    blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
    edges = cv2.Canny(blurred, 15, 80, apertureSize=3)  # Tighter thresholds
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    cv2.imwrite(f"debug_edges_{debug_idx}.jpg", edges)
    
    # Improved thresholding
    processed = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 19, 7)  # Larger block, offset
    cv2.imwrite(f"debug_adaptive_{debug_idx}.jpg", processed)
    
    # Auto-detect corners
    for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
        scaled = cv2.resize(processed, None, fx=scale, fy=scale) if scale != 1.0 else processed
        ret, corners = cv2.findChessboardCorners(scaled, (7, 7),
                                                flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                      cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                      cv2.CALIB_CB_FAST_CHECK)
        if ret and len(corners) >= 10:
            if scale != 1.0:
                corners /= scale
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                      criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            
            # Extrapolate to full 8x8
            if len(corners) < 49:
                step_x = np.mean(np.diff(corners[:, 0, 0])) if len(corners) > 1 else 128
                step_y = np.mean(np.diff(corners[:, 0, 1])) if len(corners) > 1 else 128
                start_x = min(corners[:, 0, 0])
                start_y = min(corners[:, 0, 1])
                corners_full = []
                for i in range(8):
                    for j in range(8):
                        x = start_x + j * step_x
                        y = start_y + i * step_y
                        corners_full.append([[x, y]])
                corners = np.array(corners_full)
            
            # Map squares
            squares = {}
            for i in range(8):
                for j in range(8):
                    file = chr(ord('a') + j)
                    rank = str(8 - i)
                    square_name = f"{file}{rank}"
                    idx = i * 8 + j
                    if idx < len(corners):
                        x, y = corners[idx][0]
                        squares[square_name] = (int(x), int(y))
                    else:
                        x = corners[-1][0][0] + (j-7) * 128
                        y = corners[-1][0][1] + (i-7) * 128
                        squares[square_name] = (int(x), int(y))
            print(f"Chessboard detected at scale {scale}!")
            return squares, corners
    
    # Edge-based fallback if auto-detection fails
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        if w > 300 and h > 300:  # Ensure it’s the board
            square_size = min(w, h) // 8
            squares = {}
            for i in range(8):
                for j in range(8):
                    file = chr(ord('a') + j)
                    rank = str(8 - i)
                    x_pos = x + j * square_size + square_size // 2
                    y_pos = y + i * square_size + square_size // 2
                    squares[f"{file}{rank}"] = (int(x_pos), int(y_pos))
            print("Chessboard not auto-detected! Using edge-based fallback.")
            return squares, None
    
    print("Chessboard not detected!")
    return None, None

# Detect pieces (basic color-based)
def detect_pieces(frame, squares):
    if squares is None:
        return None
    board_state = {}
    region_size = 30
    for square, (x, y) in squares.items():
        region = frame[max(0, y-region_size):y+region_size, max(0, x-region_size):x+region_size]
        if region.size == 0:
            continue
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(hsv, axis=(0, 1))
        if mean_hsv[2] < 50 or mean_hsv[2] > 200:  # Tune for your pieces
            board_state[square] = 'piece'
        else:
            board_state[square] = 'empty'
    return board_state

# Main pipeline
def main():
    best_focus = 130
    best_sharpness = 0
    
    # Find optimal focus
    print("Finding optimal focus (50-250)...")
    for focus in range(50, 251, 10):
        pipeline = init_oakd(focus)
        with dai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            sleep(0.5)
            in_rgb = q_rgb.get()
            frame = in_rgb.getCvFrame()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            print(f"Focus: {focus}, Sharpness: {sharpness:.2f}")
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_focus = focus
            if sharpness > 100:
                break
            sleep(0.1)
    
    print(f"Optimal focus: {best_focus} with sharpness {best_sharpness:.2f}")
    pipeline = init_oakd(best_focus)
    
    try:
        with dai.Device(pipeline) as device:
            q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            debug_idx = 0
            prev_state = None
            
            while True:
                in_rgb = q_rgb.get()
                frame = in_rgb.getCvFrame()
                
                # Detect chessboard
                squares, corners = detect_chessboard(frame, debug_idx)
                debug_idx += 1
                
                if squares:
                    # Visualize board
                    if corners is not None:
                        cv2.drawChessboardCorners(frame, (7, 7), corners, True)
                    for square, (x, y) in squares.items():
                        cv2.putText(frame, square, (x-10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Detect pieces
                    board_state = detect_pieces(frame, squares)
                    if board_state and prev_state and board_state != prev_state:
                        print("Board changed!")
                        for square, state in board_state.items():
                            print(f"{square}: {state}")
                
                # Show frame
                cv2.imshow("Chessboard", frame)
                if cv2.waitKey(1) == ord('q'):
                    break
                
                if squares:
                    prev_state = board_state
                
                sleep(0.1)
                
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()