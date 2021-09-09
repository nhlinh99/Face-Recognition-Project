import os
import numpy as np

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