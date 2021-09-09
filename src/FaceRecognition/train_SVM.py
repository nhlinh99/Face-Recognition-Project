"""
    arcface extract feature + linearSVC classifier
"""

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import numpy as np
import argparse
import json
import pickle
import time
import random
import os

base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def load_train_data(rate, train_emb_path):
    labels = os.listdir(train_emb_path)
    full_data = []
    full_label = []
    print("Loading training dataset....")
    for label in tqdm(labels):
        train_label_path = os.path.join(train_emb_path, label)
        files = os.listdir(train_label_path)
        for file in files:
            file_path = os.path.join(train_label_path, file)
            emb = np.load(file_path)
            emb = emb.flatten()
            full_data.append(emb)
            full_label.append(int(label))
    #   separate train and dev set
    c = list(zip(full_data, full_label))
    random.shuffle(c)
    full_data, full_label = zip(*c)

    train_data = full_data[:int(rate*len(full_data))]
    train_label = full_label[:int(rate*len(full_data))]
    dev_data = full_data[int(rate*len(full_data)):]
    dev_label = full_data[int(rate*len(full_data)):]
    #   return
    return train_data, train_label, dev_data, dev_label


def load_test_data(test_emb_path):
    labels = os.listdir(test_emb_path)
    test_data = []
    test_filename = []
    test_label = []
    print("Loading test dataset....")
    for label in tqdm(labels):
        train_label_path = os.path.join(test_emb_path, label)
        files = os.listdir(train_label_path)
        for file in files:
            file_path = os.path.join(train_label_path, file)
            emb = np.load(file_path)
            emb = emb.flatten()
            test_data.append(emb)
            test_filename.append(file_path)
            test_label.append(int(label))
            
    return test_data, test_filename, test_label


def main(args):
    #   load train data
    print("load train emb")
    train_data, train_label, _, _ = load_train_data(rate = 1.0, train_emb_path = args.dataset_path)
        
    #   load test data
    # print("load test emb")
    # test_data, test_images, test_labels = load_test_data(os.path.join(args.dataset_path, "test"))

    #   train new SVM model
    print("Train new SVM model")
    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)

    model_name = os.path.join(args.model_path, "model_facenet.pkl")
    if not os.path.exists(model_name):
        Cs = [1., 10., 15., 20., 100., 1000.]
        parameters = {'C':Cs}
        svm = LinearSVC(multi_class = 'ovr')
        clf = GridSearchCV(svm, parameters, cv=5)
        clf.fit(X = train_data, y = train_label)
        with open(model_name, "wb") as f:
            pickle.dump(clf, f)
    else:
        with open(model_name, "rb") as f:
            clf = pickle.load(f)

    print("eval on train set")
    true_pred = 0
    for data, label in zip(train_data, train_label):
        pred_label = clf.predict([data])[0]
        if pred_label == label:
            true_pred += 1
    print("accuracy on train: ", true_pred/len(train_data))

    # print("predict label for test set")
    # threshold = -0.55

    # cnt = 0
    # step = 300
    
    # print("eval on test set")
    # true_positive = 0
    # false_positive = 0
    # false_negative = 0

    # f = open("submission.csv", "w")
    # f.write("image,label,score\n")

    # while cnt*step < len(test_data):
    #     print(cnt, end = "\r")
    #     data = test_data[cnt*step: int(cnt + 1)*step]
    #     images = test_images[cnt*step: int(cnt + 1)*step]
    #     start_time = time.time()
    #     pred_label = clf.predict(data)
    #     pred_funct = clf.decision_function(data)
    #     print("Predict time:", time.time() - start_time)

    #     for i, image in enumerate(images):
    #         funct = list(pred_funct[i])
    #         index = list(np.arange(5))

    #         funct_plus = list(pred_funct[i])
    #         funct_plus.append(threshold)
    #         index_plus = list(np.arange(5))
            

    #         funct_plus, index_plus = zip(*sorted(zip(funct_plus, index_plus), key = lambda x: -x[0]))
    #         funct, index = zip(*sorted(zip(funct, index), key = lambda x: -x[0]))

    #         if index_plus[0] == test_labels[i + cnt*step]:
    #             true_positive += 1
    #         else:
    #             if index_plus[0] == 5:
    #                 false_negative += 1
    #             else:
    #                 false_positive += 1

            # res = ""
            # prob_res = ""
            # if index_plus[0] == 5:
            #     for j in range(5):
            #         res += str(index_plus[j]) + " "
            #         prob_res += str(funct_plus[j]) + " "
            # else:
            #     for j in range(4):
            #         res += str(index[j]) + " "
            #         prob_res += str(funct[j]) + " "
            #         if j == 3:
            #             res += "0 "
            #             prob_res += str(0) + " "

            
            # f.write(image + "," + res[:-1] + "," + prob_res[:-1] + "\n")
    #     cnt += 1
    # f.close()


    # print("Number TP:", true_positive)
    # print("Number FP:", false_positive)
    # print("Number FN:", false_negative)
    # precision = round(true_positive / (true_positive + false_positive), 4)
    # recall = round(true_positive / (true_positive + false_negative), 4)
    # print("Precision: ", precision)
    # print("Recall: ", recall)
    # print("F1 Score: ", 2*precision*recall/(precision + recall))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default = base_path + "/dataset/embeddings_facenet", type=str, help = "Dataset Path for training SVM (train and test)")
    parser.add_argument("--model_path", default = base_path + "/models/SVM", type=str, help = "Path of model SVM")
    args = parser.parse_args()
    main(args)