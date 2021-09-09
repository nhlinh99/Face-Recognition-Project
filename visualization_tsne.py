import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# PCA first to speed it up

def get_file(path_file):
    list_id = os.listdir(path_file)
    list_file = []
    for id in list_id:
        id_path = os.path.join(path_file, id)
        list_file_name = os.listdir(id_path)
        for name in list_file_name:
            list_file.append(os.path.join(id_path, name))

    return list_file


def get_embedding(path_embedding):
    list_id = os.listdir(path_embedding)
    list_embedding = []
    for id in list_id:
        id_path = os.path.join(path_embedding, id)
        list_embedding_name = os.listdir(id_path)
        for name in list_embedding_name:
            list_embedding.append(np.load(os.path.join(id_path, name)))

    return list_embedding


def scatter_thumbnails(data, images, zoom=0.12, colors=None):
    assert len(data) == len(images)

    # reduce embedding dimentions to 2
    x = PCA(n_components=2).fit_transform(data) if len(data[0]) > 2 else data

    # create a scatter plot.
    f = plt.figure(figsize=(22, 15))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], s=4)
    _ = ax.axis('off')
    _ = ax.axis('tight')

    # add thumbnails :)
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    for i in range(len(images)):
        image = plt.imread(images[i])
        im = OffsetImage(image, zoom=zoom)
        bboxprops = dict(edgecolor=colors[i]) if colors is not None else None
        ab = AnnotationBbox(im, x[i], xycoords='data',
                            frameon=(bboxprops is not None),
                            pad=0.02,
                            bboxprops=bboxprops)
        ax.add_artist(ab)
    

    return ax

list_embedding = get_embedding("./dataset/embededs/train")
list_img = get_file("./dataset/label")

x = PCA(n_components=50).fit_transform(list_embedding)
x = TSNE(perplexity=50,
         n_components=3).fit_transform(x)

_ = scatter_thumbnails(x, list_img, zoom=0.4)

plt.title('3D t-Distributed Stochastic Neighbor Embedding')
plt.show()