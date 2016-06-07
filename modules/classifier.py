
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier as DTF
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle
import numpy as np
import cv2
import imutils
from ChannelFeatures import ChannelFeatures
from feature_gen import FeatureGenerator
from template_generator import TemplateGenerator


class Classifier:

    def __init__(self, n_estimators=None, max_depth=None, clf=None):
        """ Instantiates adaboost classifier """
        #=====[ loading pretrained classifier object ]=====
        if clf:
	    	self.clf = pickle.load(open(clf,'rb')).clf
	    	print self.clf
        else:
        #=====[ Initialize new classifier ]=====
            self.clf = AdaBoostClassifier(base_estimator=DTF(max_depth=2), n_estimators=n_estimators)

    def train(self, X, Y):
        """ Trains classifier and prints average cross validation score """
        
        self.clf.fit(X, Y)

    def top_ft_indices(self, n):
        top_ft = self.clf.feature_importances_.argsort()
        return top_ft[::-1][:n] if n else top_ft[::-1]

	def predict(self, img_paths): 
		""" 
			Tests classifier against given image img_paths

			1) Generates templates
			2) Extracts channel features
			3) Genereates feature vectors
			4) classifies image 
		"""
	
		#=====[ Instantiate Channel Features ]=====		
		cf = ChannelFeatures()

		#=====[ Instantiate FeatureGenerator ]=====
		fg = FeatureGenerator(templates)
		
		#=====[ Will hold our generated feature vectors ]=====
		feature_vectors = []
		
		print '-----> Testing %d total images' % (len(img_paths))
		for idx, img_path in enumerate(img_paths):
			img = cv2.imread(img_path)
			#=====[ Extract channel features from images and make feature vector ]=====
			cfeats = cf.compute_channels(img)
			feature_vectors.append(fg.generate_features(cfeats))

			if idx % 100 == 0:
				print '-----> Processing Image ', idx + 1
		
		print '-----> Processed all feature vectors'		

		#=====[ predict class for each feature_vector ]=====
		ys = self.clf.predict(feature_vectors)

		return ys

	def test(self, img_paths, Y):
		""" 
			Test accuracy against provided image paths and Y.
		"""
		ys = self.predict(img_paths)

		accuracy = accuracy_score(Y, ys)
		return accuracy
