import os
import cv2
import math
import numpy as np
import argparse
from yolov3_n_SAN.detect import yolov3_detect
from yolov3_n_SAN.san_eval import san_eval
from Coord_tools import SI_calculate


def parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weight', nargs='+', type=str,
                        default='./yolov3_n_SAN/weights/yolov3_widerface.pt',
                        help='model path of yolov3')
    parser.add_argument('--SAN_weight', nargs='+', type=str,
                        default='./yolov3_n_SAN/weights/SAN.tar',
                        help='model path of SAN')
    parser.add_argument('--image', nargs='+', type=str,
                        default='./yolov3_n_SAN/data/images/test_3.jpg',
                        help='image path')
    paras = parser.parse_args()
    return paras


def get_file(path):
    dirs = []
    for root, dirs, files in os.walk(path):
        for file in files:
            dirs.append(os.path.join(root, file))
    print(dirs)
    return dirs


# def yolo_detect(paras):
#     """
#     Run the YOLO model on the image
#     """
#     yolo_weight = str(paras.yolo_weight)
#     image_path = str(paras.image)
#     # image_name = image_path.split('/')[-1]
#     save_dir, num_detect = yolov3_detect(image_path, yolo_weight)
#     crop_dir = get_file(f'{save_dir}/crops/face')
#     crop_dir = crop_dir[int(input(f'1~{len(crop_dir)}')) - 1]
#     img = cv2.imread(crop_dir)  # read face image
#     h, w = img.shape[0:2]
#     h_re, w_re = int(h * 640 / h), int(w * 640 / h)  # x=>640, scale y correspondingly
#     img_resize = cv2.resize(img, dsize=(w_re, h_re), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)  # scale image to x=640
#     cv2.imwrite(crop_dir, img_resize)  # save scaled image for SAN
#     return crop_dir


def yolo_detect(paras):
    """
    Run the YOLO model on the image
    """
    yolo_weight = str(paras.yolo_weight)
    image_path = str(paras.image)
    # image_name = image_path.split('/')[-1]
    results = yolov3_detect(image_path, yolo_weight)
    return results


def select_faces(yolo_dir):
    if len(yolo_dir) > 1:
        save_dir, num_detect = yolo_dir[:]
    else:
        save_dir, num_detect = yolo_dir, 0
    print(f'{num_detect} faces detected!')
    crop_dir = get_file(f'{save_dir}/crops/face')
    crop_dir = crop_dir[int(input(f'1~{len(crop_dir)}')) - 1]
    img = cv2.imread(crop_dir)  # read face image
    h, w = img.shape[0:2]
    h_re, w_re = int(h * 640 / h), int(w * 640 / h)  # x=>640, scale y correspondingly
    img_resize = cv2.resize(img, dsize=(w_re, h_re), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)  # scale image to x=640
    cv2.imwrite(crop_dir, img_resize)  # save scaled image for SAN
    return crop_dir


def SAN_detect(crop_dir, paras):
    """
    Run the SAN model on the cropped detected face image
    """
    san_dir = crop_dir.split('.')[-2] + '_san.' + crop_dir.split('.')[-1]  # image save path for SAN
    SAN_weight = str(paras.SAN_weight)  # read weight file path of SAN from arguments
    img = cv2.imread(crop_dir)  # read face image
    h, w = img.shape[0:2]
    boxes = (0, 0, w, h)  # size of image is the size of box for SAN
    landmarks, confidence = san_eval(crop_dir, SAN_weight, boxes, san_dir)  # evaluate image
    return san_dir, landmarks


def landmark_normalize(image_dir, landmarks):
    """
    Resize, Rotate, Stretch to a normalized size
    Image employing:
        cv2.getAffineTransform() or cv2.getRotationMatrix2D() --> cv2.warpAffine()
    Landmarks employing:
        def mapping_landmarks()
    """
    img = cv2.imread(image_dir)  # read image
    h, w = img.shape[0:2]  # Original height and width
    h_re, w_re = 640, 640  # Expected size
    edge, edge_resize = np.float32([[0, 0], [0, h], [w, 0]]), np.float32([[0, 0], [0, h_re], [w_re, 0]])
    M_resize = cv2.getAffineTransform(edge, edge_resize)
    img_resize = cv2.warpAffine(img, M_resize, (w_re, h_re))
    landmarks_resize = mapping_landmarks(landmarks, M_resize)
    face_left, face_right = landmarks_resize[0], landmarks_resize[16]  # rotating image
    rotate_center = (w_re / 2, h_re / 2)
    angle = math.atan2(face_left[1] - face_right[1], face_left[0] - face_right[0])  # calculate rotation angle
    angle = angle * 180 / math.pi
    angle = angle if abs(angle) < 90 else angle + 180
    M_rotate = cv2.getRotationMatrix2D(rotate_center, angle, 1.0)  # Rotation matrix
    img_rotate = cv2.warpAffine(img_resize, M_rotate, (w_re, h_re))  # rotate
    landmarks_rotate = mapping_landmarks(landmarks_resize, M_rotate)
    # stretching image
    face_left, face_right = (landmarks_rotate[0] + landmarks_rotate[1] + landmarks_rotate[3]) / 3, (
            landmarks_rotate[16] + landmarks_rotate[15] + landmarks_rotate[14]) / 3
    chin, mbrow = landmarks_rotate[8], (landmarks_rotate[19] + landmarks_rotate[24]) / 2
    pts = np.float32([[face_left[0], chin[1]], [face_right[0], chin[1]], mbrow])
    pts_stretch = np.float32(
        [[w_re * 0.1, h_re * 0.9], [w_re * 0.9, h_re * 0.9], [mbrow[0], h_re * 0.1]])
    M_stretch = cv2.getAffineTransform(pts, pts_stretch)
    img_normal = cv2.warpAffine(img_rotate, M_stretch, (w_re, h_re))
    landmarks_normal = mapping_landmarks(landmarks_rotate, M_stretch)
    # save image
    save_dir = image_dir.split('.')[0] + '_normalized.jpg'  # save path
    cv2.imwrite(save_dir, img_normal)
    # landmarks_normal[:, 0], landmarks_normal[:, 1] = landmarks_normal[:, 0] / w_re, landmarks_normal[:, 1] / h_re
    return save_dir, landmarks_normal


def mapping_landmarks(pts, trans_metrix):
    """
    Mapping landmarks using corroding transformation matrix
    An utils for def landmarks_normalize()
    """
    pts = np.float32(pts).reshape([-1, 2])
    pts = np.hstack([pts, np.ones([len(pts), 1])]).T
    new_pts = np.dot(trans_metrix, pts)
    new_pts = np.array([[new_pts[0][x], new_pts[1][x]] for x in range(len(new_pts[0]))])
    return new_pts


if __name__ == '__main__':
    paras = parameters()
    crop_dir = yolo_detect(paras)
    crop_dir = select_faces(crop_dir)
    san_dir, landmarks = SAN_detect(crop_dir, paras)
    save_dir, landmarks_normal = landmark_normalize(san_dir, landmarks)
    SI_data = SI_calculate(save_dir, landmarks_normal)
    # print('SI_Eyebrows = %d%%, SI_Eyes = %d%%, SI_mouth = %d%%' % (SI_Eyebrows, SI_Eyes, SI_Mouth))
