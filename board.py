import cv2
import numpy as np
import depthai as dai

# Initialize Oak-D Lite
pipeline = dai.Pipeline()
cam_rgb = pipeline.create(dai.node.ColorCamera)
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.setPreviewSize(1024, 1024)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)
cam_rgb.initialControl.setManualFocus(150)  # Adjust based on sharpness
cam_rgb.preview.link(xout_rgb.input)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    while True:
        in_rgb = q_rgb.get()
        frame = in_rgb.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        print(f"Sharpness: {sharpness:.2f}")

        # Board detection (simple edge-based)
        edges = cv2.Canny(gray, 15, 80)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            if w > 300 and h > 300:  # Board size check
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                board_rect = (x, y, w, h)
                print("Board detected!")
                break
        cv2.imshow("Board Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    if 'board_rect' in locals():
        return board_rect