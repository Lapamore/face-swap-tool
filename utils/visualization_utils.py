import cv2

def draw_landmarks(img, triangle_list):
    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        
        cv2.line(img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA)

    return img
