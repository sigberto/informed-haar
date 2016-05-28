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

		for t in self.templates:

			x, y, size, W = t
			w,h = size
			
			for k in range(k_channels):
				
				cell_feats = np.copy(cfeats[y:y+h, x:x+w, k])
				self.features.append(np.sum(np.multiply(cell_feats,W)))
				self.feature_info.append((x,y,size,k))
				
		return self.features