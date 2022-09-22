import numpy as np
import cv2

def draw_keypoints(image, joints, width, height, color=(245,117,66), thickness=2, circle_radius=2):
    for i in range(13):
        x = joints[0][i]
        y = joints[1][i]
        cv2.circle(image, (int(x*width),int(y*height)), circle_radius, color, thickness)
    # return cv2.addWeighted(overlay, 1.0, image, 0.0, 0)
    return image

def _draw_connection(image, point1, point2, color=(245,66,230), thickness=2):
    x1, y1 = point1
    x2, y2 = point2
    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, cv2.LINE_AA)
    # print('==========')
    return image

def draw_connection(image, joints, width, height):
    # overlay = image.copy()
    connection= [(0,1), (0,3), (1,2), (3,4), (4,5), (0,12), (3,12), (0,6),
                (3,9), (6,7), (7,8), (9,10), (10,11), (6,9)]
    kpts = []
    for i in range(13):
        x = joints[0][i]
        y = joints[1][i]
        x *= width
        y *= height
        kpts.append([x,y])
    for conn in connection:
        x,y = conn
        image = _draw_connection(image, kpts[x], kpts[y])
    # return cv2.addWeighted(overlay, 1.0, image, 0.0, 0)
    return image