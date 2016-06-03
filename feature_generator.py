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
	    try:
	    	temp = t[0]
           	x = temp[0]
	 	y = temp[1]
	    	size  = temp[2] 
	    	W = temp[3]
            	w, h = size
            	k = t[1]
	    except Exception as e:
		print e
		print 'template: ', t
            cell_feats = cfeats[y:y + h, x:x + w, k]
            self.features.append(np.sum(np.multiply(cell_feats, W)))            

        return self.features
