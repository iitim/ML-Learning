import os

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, homogeneity_score
import numpy as np

from util.image_loader import load_image


dts_dir = "../formatted_img_dataset"
train_dir = os.path.join(dts_dir, 'train')
val_dir = os.path.join(dts_dir, 'validation')
test_dir = os.path.join(dts_dir, 'test')

train_loader, val_loader, test_loader = load_image(train_dir, val_dir, test_dir)


images, labels = iter(train_loader).__next__()
np_inputs = images.numpy()
train = np_inputs.reshape(len(np_inputs), -1)

kmeans = KMeans(n_clusters=2, max_iter=1000, n_init=20)
kmeans.fit(train)


def calculate_metrics(model, output):
    print('Number of clusters is {}'.format(model.n_clusters))
    print('Inertia : {}'.format(model.inertia_))
    print('Accuracy : {}'.format(accuracy_score(output, model.labels_)))
    print('Homogeneity : {}'.format(homogeneity_score(output, model.labels_)))


calculate_metrics(kmeans, labels.numpy())
