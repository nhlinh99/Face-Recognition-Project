#############################
# Modefied inference of mxnet
#https://github.com/deepinsight/insightface/issues/1417#issue-819308039
#https://github.com/deepinsight/insightface/issues/1417
#no tensor normalize,see https://github.com/deepinsight/insightface/issues/1417#issuecomment-894589411
#############################
# -*- coding: utf-8 -*-
import numpy as np
import cv2
import mxnet as mx
from collections import namedtuple
import time
import os

#import pkg_resources
#print("mxnet version:", pkg_resources.get_distribution("mxnet").version)

def create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)


# convert array
def get_array(face_chip):
    face_chip = cv2.cvtColor(face_chip, cv2.COLOR_BGR2RGB)
    face_chip = face_chip.transpose(2, 0, 1)
    face_chip = face_chip[np.newaxis, :] # 4d
    array = mx.nd.array(face_chip)
    return array

# load mxnet weight
prefix = "./models/Resnet100/model"
sym, arg, aux = mx.model.load_checkpoint(prefix, 0)

# define mxnet
ctx = mx.gpu(0) # gpu_id = 0 ctx = mx.gpu(gpu_id)
mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,112,112))])
mod.set_params(arg, aux)
Batch = namedtuple('Batch', ['data'])

output_path = "/home/louis/Desktop/output_embededs/"
create_dir(output_path)
folder_path = "/home/louis/Desktop/align_images/"
list_id = os.listdir(folder_path)


for id in list_id:

    create_dir(os.path.join(output_path, id))

    img_list = [os.path.join(folder_path, id, f) for f in os.listdir(os.path.join(folder_path, id))]

    for im_path in img_list:
        im_name = os.path.basename(im_path)
        # read image
        img = cv2.imread(im_path)
        array = get_array(img)

        # inference
        start_time = time.time()
        mod.forward(Batch([array]))

        # feature
        feat = mod.get_outputs()[0].asnumpy()
        print("Inference time for {}:".format(im_name), time.time() - start_time)
        np.save(os.path.join(output_path, id, im_name).replace(".jpg", ".npy"), feat[0])
