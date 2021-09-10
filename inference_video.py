import cv2
import time
import pickle
import numpy as np
import mxnet as mx
from collections import namedtuple

from src.FaceDetection import face_inference, retinaface
from src.FaceRecognition import inference_face_embedding, inference_SVM, compare_similarity
from get_embedding_db import get_embedding_db


detection_threshold = 0.85
classification_threshold = 0 #-0.55

gpuid = 0
detector = retinaface.RetinaFace('./models/detection/Mobilenet/mnet.25', 0, gpuid, 'net3')
# detector = retinaface.RetinaFace('./models/detection/Resnet50/R50', 0, gpuid, 'net3')

# load mxnet weight
prefix = "./models/recognition/MFaceNet/model"
sym, arg, aux = mx.model.load_checkpoint(prefix, 0)

# define mxnet
ctx = mx.gpu(0) # gpu_id = 0 ctx = mx.gpu(gpu_id)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
mod.set_params(arg, aux)
batch = namedtuple('Batch', ['data'])

# Load SVM model
with open("./models/SVM/model_facenet.pkl", "rb") as f:
    SVM_model = pickle.load(f)


list_label, list_embeddings = get_embedding_db("./dataset/embeddings_facenet")

cap = cv2.VideoCapture('./dataset/Security_Camera.mp4')
scales = [720, 1280]
font = cv2.FONT_HERSHEY_SIMPLEX
    
prev_frame_time = 0
new_frame_time = 0

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_video = cv2.VideoWriter('./output/result_Facenet_cosine_sim.avi', fourcc, 20.0, (640,360))

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
            face_embeded = inference_face_embedding.get_face_embeded(face, mod, batch)
            list_embeddeds.append(face_embeded)

        for embedding_face in list_embeddeds:
            # cosine_similarity = compare_similarity.get_cosine_similarity(embedding_face, list_embeddings, list_label)
            # identity = compare_similarity.get_target_id(cosine_similarity)
            identity = inference_SVM.inference_identity(embedding_face, SVM_model, classification_threshold)
            list_identities.append(identity)

        for i in range(faces.shape[0]):
            box = faces[i].astype(np.int32)
            identity = list_identities[i]
            color = (0, 0, 255)
            print(box)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(frame, str(identity), (box[0], box[1] - 7), font, 1, (100, 255, 0), 2, cv2.LINE_AA)

    
    # font which we will be using to display FPS
    # time when we finish processing for this frame
    new_frame_time = time.time()

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)
    fps = "FPS: " + str(fps)

    frame = cv2.resize(frame, (640, 360))

    # putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 40), font, 1, (100, 255, 0), 2, cv2.LINE_AA)

    # output_video.write(frame)
    # displaying the frame with fps
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
# output_video.release()
# Destroy the all windows now
cv2.destroyAllWindows()