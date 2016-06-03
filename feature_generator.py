import numpy as np


class FeatureGenerator:
    def __init__(self, templates):
        """
            Instantiates feature generator with lists to store feature vectors and corresponding template
            information for each feature vector
        """

        self.templates = templates
        self.feature_info = []

    def generate_features(self, cfeats):
        """ Generates feature vectors associated with each template """

        self.features = []

        for indx, t in enumerate(self.templates):

            x, y, size, W = t[1]
            w, h = size
            k = t[0]

            cell_feats = cfeats[y:y + h, x:x + w, k]
            self.features.append(np.sum(np.multiply(cell_feats, W)))            

        return self.features
