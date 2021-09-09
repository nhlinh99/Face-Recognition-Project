import cv2
import numpy as np
import time
import os

from retinaface import RetinaFace
from utils.alignment import get_reference_facial_points, warp_and_crop_face


base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

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


thresh = 0.8

gpuid = 0
# detector = RetinaFace('./models/detection/Mobilenet/mnet.25', 0, gpuid, 'net3')
detector = RetinaFace(base_path + '/models/detection/Resnet50/R50', 0, gpuid, 'net3')
dirname = base_path + "/dataset/label"
create_dir(dirname)

cap = cv2.VideoCapture(base_path + '/dataset/Security_Camera.mp4')

# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

count = 0

while(cap.isOpened()):

    count += 1

    ret, frame = cap.read()
    # if video finished or no Video Input
    if not ret:
        break


    scales = [405, 720]
    im_shape = frame.shape
    target_size = scales[0]
    max_size = scales[1]
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    
    im_scale = float(target_size) / float(im_size_min)
    
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)


    scales = [im_scale]
    flip = False

    faces, landmarks = detector.detect(frame,
                                    thresh,
                                    scales=scales,
                                    do_flip=flip)


    facial_5_points_list = landmarks
    if (count % 10 == 0):
        k = 0
        for facial_5_points in facial_5_points_list:
            dst_img = process(frame, facial_5_points, (112,112))

            cv2.imwrite(os.path.join(dirname, "{}_{}".format(count, k) + ".jpg"), dst_img)
            k += 1


    if faces is not None:
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int32)
            color = (0, 0, 255)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)

            if landmarks is not None:
                landmark5 = landmarks[i].astype(np.int32)
                for l in range(landmark5.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(frame, (landmark5[l][0], landmark5[l][1]), 1, color, 2)

    
    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = "FPS: " + str(fps)

    # putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 40), font, 1, (100, 255, 0), 2, cv2.LINE_AA)

    frame = cv2.resize(frame, (720, 405))
    # displaying the frame with fps
    cv2.imshow('frame', frame)

    # press 'Q' if you want to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
# Destroy the all windows now
cv2.destroyAllWindows()