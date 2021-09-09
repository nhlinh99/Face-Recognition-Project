import numpy as np
import os
import time


upper_threshold = 0.8
lower_threshold = 0.2

confidence_cosine_threshold = 0.5
priority_for_highest_sim = 3


def get_embedding_db(embedding_path):
    list_embeddings = []
    list_label = []

    set_label = os.listdir(embedding_path)
    for label in set_label:
        label_path = os.path.join(embedding_path, label)
        list_emb_name = os.listdir(label_path)
        for emb_name in list_emb_name:
            list_embeddings.append(np.load(os.path.join(label_path, emb_name)).tolist())
            list_label.append(label)

    return list_label, np.array(list_embeddings)


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return np.divide(v, norm)


def normalize_matrix(m):
    norm = np.linalg.norm(m, axis = 1)
    if 0 in norm:
       return m
    return np.divide(m.T, norm).T 


def get_cosine_similarity(compared_embedding, list_embeddings, list_label):

    # list_similarity = normalize_matrix(list_embeddings) @ normalize_vector(compared_embedding)
    list_similarity = normalize_matrix(list_embeddings).dot(normalize_vector(compared_embedding))
    list_label_similarity = np.column_stack((list_label, list_similarity.astype(np.object)))
    list_label_similarity = list_label_similarity[np.lexsort((list_label_similarity[:,0],list_label_similarity[:,1]))][::-1]
    return list_label_similarity


def get_target_id(list_similarity):
    top_10_similarity = list_similarity[:10]
    # print(top_10_similarity)
    if (list_similarity[0][1] >= upper_threshold):
        return list_similarity[0][0]
    elif (list_similarity[0][1] <= lower_threshold):
        return 0

    set_top_label = set([sim[0] for sim in top_10_similarity])
    dict_similarity = {label: 0 for label in set_top_label}
    
    dict_similarity[top_10_similarity[0][0]] += top_10_similarity[0][1] * priority_for_highest_sim
    for i in range(1, len(top_10_similarity)):
        dict_similarity[top_10_similarity[i][0]] += top_10_similarity[i][1]

    list_similarity_filter = sorted(dict_similarity.items(), key=lambda kv: -kv[1])
    if (list_similarity_filter[0][1] >= confidence_cosine_threshold * priority_for_highest_sim):
        return list_similarity_filter[0][0]
    else:
        return 0


if __name__=="__main__":
    base_path = "/home/louis/Desktop/Face Recognition Project/dataset/embeddings_facenet"

    list_label, list_embeddings = get_embedding_db(base_path)

    compared_embedding = np.load("/home/louis/Desktop/Face Recognition Project/dataset/embeddings_facenet/2/730_2.npy").tolist()
    start = time.time()
    list_similarity = get_cosine_similarity(compared_embedding, list_embeddings, list_label)
    print("Inference time:", time.time() - start)
    start = time.time()
    target_id = get_target_id(list_similarity)
    print(target_id)
    print("Inference time:", time.time() - start)