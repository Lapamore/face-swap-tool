import cv2
import numpy as np
import argparse
import logging 
from utils.face_landmarks import get_landmarks_and_triangles
from utils.find_triangle_indices_utils import find_triangle_indices


class FaceSwapper():
    def __init__(self, img_source_path):
        self.img_source_path = img_source_path

    def read_and_resize_images(self):
        self.img_source = cv2.imread(self.img_source_path)
        self.img_source = cv2.resize(self.img_source, (self.img_source.shape[1] // 2, self.img_source.shape[0] // 2))
        self.img_source_gray = cv2.cvtColor(self.img_source, cv2.COLOR_BGR2GRAY)

    def process_images(self):
        cap = cv2.VideoCapture(0) 

        while True:
            _, img_destination = cap.read()
            img_destination = cv2.resize(img_destination, (img_destination.shape[1] // 4, img_destination.shape[0] // 4))
            img_destination_gray = cv2.cvtColor(img_destination, cv2.COLOR_BGR2GRAY)    
            img_face = np.zeros(img_destination.shape, np.uint8) 
            img_face_mask = np.zeros_like(img_destination_gray)

            landmarks_source, triangle_source, _ = get_landmarks_and_triangles(self.img_source_gray)
            landmarks_destination, _, convexhull2 = get_landmarks_and_triangles(img_destination_gray)

            triangles_indeces = find_triangle_indices(triangle_source, landmarks_source)

            for index in triangles_indeces:
                # Первое лицо
                pt1 = landmarks_source[index[0]]
                pt2 = landmarks_source[index[1]]
                pt3 = landmarks_source[index[2]]
                triangle1 = np.array([pt1, pt2, pt3], np.int32)

                (x, y, w, h) = cv2.boundingRect(triangle1)
                cropped_triangle = self.img_source[y: y + h, x: x + w]

                points = np.array([[pt1[0] - x, pt1[1] - y],
                                [pt2[0] - x, pt2[1] - y],
                                [pt3[0] - x, pt3[1] - y]], np.int32)

                # Второе лицо
                pt1_ = landmarks_destination[index[0]]
                pt2_ = landmarks_destination[index[1]]
                pt3_ = landmarks_destination[index[2]]
                triangle2 = np.array([pt1_, pt2_, pt3_], np.int32)

                (x, y, w, h) = cv2.boundingRect(triangle2)

                cropped_tr2_mask = np.zeros((h, w), np.uint8)

                points2 = np.array([[pt1_[0] - x, pt1_[1] - y],
                                    [pt2_[0] - x, pt2_[1] - y],
                                    [pt3_[0] - x, pt3_[1] - y]], np.int32)

                cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

                # int32 -> float32
                points = np.float32(points)
                points2 = np.float32(points2)

                M = cv2.getAffineTransform(points, points2)
                warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

                img_new_face_area = img_face[y: y + h, x: x + w]
                img_new_face_area_gray = cv2.cvtColor(img_new_face_area, cv2.COLOR_BGR2GRAY)
                _, mask_triangles = cv2.threshold(img_new_face_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
                warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles)

                img_new_face_area = cv2.add(img_new_face_area, warped_triangle)
                img_face[y: y + h, x: x + w] = img_new_face_area

            img_head_mask = cv2.fillConvexPoly(img_face_mask, convexhull2, 255)
            img_face_mask = cv2.bitwise_not(img_head_mask)

            img_without_face = cv2.bitwise_and(img_destination, img_destination, mask=img_face_mask)
            result_img = cv2.add(img_without_face, img_face)

            (x, y, w, h) = cv2.boundingRect(convexhull2)
            center = (int((x + x + w) / 2), int((y + y + h) / 2))

            self.final_img = cv2.seamlessClone(result_img, 
                                            img_destination,
                                            img_head_mask, 
                                            center, 
                                            cv2.NORMAL_CLONE)
            
            self.display_result()
            if cv2.waitKey(1) == ord('q'):
                break
        
    def display_result(self):
        cv2.imshow('FaceSwap Image', self.final_img)
        
    def run(self):
        self.read_and_resize_images()
        self.process_images()

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Программа для замены лица на видео.')
    parser.add_argument('--source', type=str, required=True, help='Путь к изображению, с которого будет взято лицо')

    args = parser.parse_args()

    face_swapper = FaceSwapper(img_source_path=args.source)
    try:
        face_swapper.run()
    except Exception as e:
        logger.error(f"Произошла ошибка: {e}")

if __name__ == '__main__':
    main()
