import dlib
import cv2
import numpy as np

def load_models():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

    return detector, predictor

def get_landmarks_and_convexhull(img_main):
    landmarks = []
    detector, predictor = load_models()
    img_main_detected_boxes = detector(img_main)
    for box in img_main_detected_boxes:
        shape = predictor(img_main, box)
        for i in range(68):
            landmarks.append((shape.part(i).x, shape.part(i).y))   
        points = np.array(landmarks, np.int32)
        convexhull = cv2.convexHull(points)

    return landmarks, convexhull

def get_landmarks_and_triangles(img_main):
    landmarks, convexhull =  get_landmarks_and_convexhull(img_main)
    size = img_main.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(landmarks)
    triangle_list = subdiv.getTriangleList()

    return np.array((landmarks), np.int32), np.array((triangle_list), np.int32), convexhull   
