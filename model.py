import depthai as dai
import cv2

# Create pipeline
pipeline = dai.Pipeline()

# Define a source (e.g., RGB camera)
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

# Create output
xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

# Connect to device and run
with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb")
    while True:
        frame = q_rgb.get().getCvFrame()
        cv2.imshow("RGB Preview", frame)
        if cv2.waitKey(1) == ord('q'):
            break