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
        self.feature_info = []

        _, _, k_channels = cfeats.shape

        for indx, t in enumerate(self.templates):

            #=====[ Get channel and template from (channel, template) tuple t
            k = t[1]
            t = t[0]

            x, y, size, W = t
            w, h = size

            cell_feats = np.copy(cfeats[y:y + h, x:x + w, k])
            #=====[ Multiply channel features by template weight matrix W and sum ]=====
            self.features.append(np.sum(np.multiply(cell_feats, W)))

        return self.features
