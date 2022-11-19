import pandas as pd
import logging

from u_k_means_model import UKMeans


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    blobs = pd.read_csv('../k_means_clustering/dataset/kmeans_blobs.csv')
    dataset = blobs[['x', 'y']].values.tolist()
    model = UKMeans(3, dataset)
    model.main()


if __name__ == '__main__':
    main()
