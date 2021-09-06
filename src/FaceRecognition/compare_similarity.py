import numpy as np
import os

input_path = "/home/louis/Desktop/output_embededs/C_500_0.npy"
input_embedding = np.load(input_path)

base_path = "/home/louis/Desktop/output_embededs/"
list_id = ["A", "B", "C", "D", "E"]

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


list_similarity = []
list_id_name = []

for id in list_id:
    list_embeddings_name = os.listdir(os.path.join(base_path, id))
    for name in list_embeddings_name:
        compared_embedding = np.load(os.path.join(base_path, id, name))

        embedding_1, embedding_2 = normalize(input_embedding), normalize(compared_embedding)
        sim = embedding_1 @ embedding_2.T
        print("Compare {} with id {}:".format(os.path.basename(input_path), id + "/" + name), sim)

        list_similarity.append(sim)
        list_id_name.append(id)

list_similarity = sorted(zip(list_similarity, list_id_name), key = lambda x: -x[0])
print(list_similarity)