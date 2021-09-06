import numpy as np
import os
import sys

sys.path.append("./src/FaceDetection")
from utils.alignment import get_reference_facial_points, warp_and_crop_face


def create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


def process(img, facial_5_points, output_size):

    facial_points = np.array(facial_5_points)

    default_square = True
    inner_padding_factor = 0.5
    outer_padding = (0, 0)

    # get the reference 5 landmarks position in the crop settings
    reference_5pts = get_reference_facial_points(
        output_size, inner_padding_factor, outer_padding, default_square)

    dst_img = warp_and_crop_face(img, facial_points, reference_pts=reference_5pts, crop_size=output_size)
    
    return dst_img


def get_face_area(img, detector, threshold, scales = [640, 1200]):

    im_shape = img.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    
    im_scale = float(target_size) / float(im_size_min)
    
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    scales = [im_scale]
    flip = False

    faces, landmarks = detector.detect(img,
                                    threshold,
                                    scales=scales,
                                    do_flip=flip)

    crop_faces = []
    for facial_5_points in landmarks:
        crop_face_img = process(img, facial_5_points, (112,112))
        crop_faces.append(crop_face_img)

    return crop_faces, faces, landmarks