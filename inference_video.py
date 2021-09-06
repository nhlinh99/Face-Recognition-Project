import cv2
import time
import pickle
import numpy as np
import mxnet as mx
from collections import namedtuple

from src.FaceDetection import face_inference, retinaface
from src.FaceRecognition import inference_arcface, inference_SVM


detection_threshold = 0.9
classification_threshold = -0.55

gpuid = 0
detector = retinaface.RetinaFace('./models/detection/Mobilenet/mnet.25', 0, gpuid, 'net3')
# detector = RetinaFace('./models/detection/Resnet50/R50', 0, gpuid, 'net3')

# load mxnet weight
prefix = "./models/recognition/Resnet100/model"
sym, arg, aux = mx.model.load_checkpoint(prefix, 0)

# define mxnet
ctx = mx.gpu(0) # gpu_id = 0 ctx = mx.gpu(gpu_id)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
mod.set_params(arg, aux)
batch = namedtuple('Batch', ['data'])

# Load SVM model
with open("./models/SVM/model.pkl", "rb") as f:
    SVM_model = pickle.load(f)

cap = cv2.VideoCapture('./dataset/Security_Camera.mp4')
scales = [480, 900]
    
prev_frame_time = 0
new_frame_time = 0

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video = cv2.VideoWriter('./output/result.avi', fourcc, 20.0, (900,480))

while(cap.isOpened()):

    ret, frame = cap.read()
    # if video finished or no Video Input
    if not ret:
        break

    crop_faces, faces, landmarks = face_inference.get_face_area(frame, detector, detection_threshold, scales)

    list_embeddeds = []
    list_identities = []
    if (len(crop_faces) > 0):
        for face in crop_faces:
            face_embeded = inference_arcface.get_face_embeded(face, mod, batch)
            list_embeddeds.append(face_embeded)

        for embeded_face in list_embeddeds:
            identity = inference_SVM.inference_identity(embeded_face, SVM_model, classification_threshold)
            list_identities.append(identity)


    if faces is not None:
        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int32)
            identity = list_identities[i]
            color = (0, 0, 255)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, str(identity), (box[0], box[1] - 7), font, 1, (100, 255, 0), 2, cv2.LINE_AA)

    
    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)
    fps = "FPS: " + str(fps)

    frame = cv2.resize(frame, (900, 480))

    # putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 40), font, 1, (100, 255, 0), 2, cv2.LINE_AA)

    output_video.write(frame)
    # displaying the frame with fps
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
output_video.release()
# Destroy the all windows now
cv2.destroyAllWindows()