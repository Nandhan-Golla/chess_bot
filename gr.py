import cv2
import numpy as np
import depthai as dai
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *  # Added GLUT import
import pygame
from pygame.locals import *

# Initialize Oak-D Lite
def init_oakd():
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
    return pipeline

# Draw 3D chessboard
def draw_chessboard():
    glBegin(GL_QUADS)
    for i in range(8):
        for j in range(8):
            glColor3fv([0.5 if (i+j)%2==0 else 1.0, 0.5 if (i+j)%2==0 else 1.0, 0.5])
            glVertex3f(i-4, j-4, 0)
            glVertex3f(i-4+1, j-4, 0)
            glVertex3f(i-4+1, j-4+1, 0)
            glVertex3f(i-4, j-4+1, 0)
    glEnd()

# Draw 3D pieces
def draw_pieces(pieces):
    for (i, j), _ in pieces.items():
        glPushMatrix()
        glTranslatef(i-3.5, j-3.5, 1.0)  # Place piece
        glColor3f(1, 0, 0)  # Red piece (simplified)
        glutSolidSphere(0.4, 20, 20)  # Now works with GLUT
        glPopMatrix()

# Initialize GLUT (required for glutSolidSphere)
glutInit()

with dai.Device(init_oakd()) as device:
    q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    pygame.init()
    display = (1024, 1024)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)

    board_rect = None
    while True:
        in_rgb = q_rgb.get()
        frame = in_rgb.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Board detection
        if board_rect is None:
            edges = cv2.Canny(gray, 15, 80)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                if w > 300 and h > 300:
                    board_rect = (x, y, w, h)
                    print("Board detected!")

        # Piece detection
        pieces = {}
        if board_rect:
            x, y, w, h = board_rect
            square_size = w // 8
            for i in range(8):
                for j in range(8):
                    region = frame[y+i*square_size:y+(i+1)*square_size, x+j*square_size:x+(j+1)*square_size]
                    if region.size == 0:
                        continue
                    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
                    mean_hsv = np.mean(hsv, axis=(0, 1))
                    if mean_hsv[2] < 50 or mean_hsv[2] > 200:  # Piece threshold
                        pieces[(i, j)] = 'piece'

        # Render 3D
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        draw_chessboard()
        if pieces:
            draw_pieces(pieces)
        pygame.display.flip()

        # Overlay on frame
        frame_with_ar = frame.copy()
        if board_rect:
            x, y, w, h = board_rect
            cv2.rectangle(frame_with_ar, (x, y), (x+w, y+h), (0, 255, 0), 2)
            for (i, j), _ in pieces.items():
                cx = x + j * square_size + square_size // 2
                cy = y + i * square_size + square_size // 2
                cv2.circle(frame_with_ar, (cx, cy), 10, (0, 0, 255), -1)
        cv2.imshow("AR Chessboard", frame_with_ar)
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()
    pygame.quit()