<<<<<<< HEAD
from sklearn.datasets import load_iris
=======
>>>>>>> evaluation
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
from feature_generator import FeatureGenerator
from template_generator import TemplateGenerator


class Classifier:

	def __init__(self, n_estimators=None, max_depth=None, clf=None):
		""" Instantiates adaboost classifier """

		#=====[ loading pretrained classifier object ]=====
		if clf:
			self.clf = pickle.load(open('BoostedTreeclassifier_small.p','rb')).clf
			print self.clf
		else:
		#=====[ Initialize new classifier ]=====
			self.clf = AdaBoostClassifier(base_estimator=DTF(max_depth=2), n_estimators=n_estimators)

    def train(self, X, Y):
        """ Trains classifier and prints average cross validation score """

        # scores = cross_val_score(self.clf, X, Y)
        # print "Average Cross ValidationScore: %s" % ("{0:.2f}".format(scores.mean()))
        self.clf.fit(X, Y)

    def top_ft_indices(self, n):
        top_ft = self.clf.feature_importances_.argsort()
        return top_ft[::-1][:n] if n else top_ft[::-1]

    def plot_ft_weights(self, file_name):
        # Get corresponding template information for each feature
        feature_info = pickle.load(open('feature_info.p', 'r'))
        w_viz = np.zeros((20, 10))

        # Sum weights from each feature corresponding to cells in our templates
        for idx, w in enumerate(self.clf.feature_importances_):
            x, y, size, k = feature_info[idx]
            w, h = size
            w_viz[y:y + h, x:x + w] += w

        # Normalize weight visualization matrix and display
        w_viz = w_viz / np.max(w_viz)
        im = plt.matshow(w_viz, cmap='Reds')
        plt.title('Feature Importances')

        ax = plt.gca()

         # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        plt.savefig(file_name)


	def predict(self, img_paths): 
		""" 
			Tests classifier against given image img_paths

			1) Generates templates
			2) Extracts channel features
			3) Genereates feature vectors
			4) classifies image 
		"""
	
		#=====[ Generate templates ]=====		
		tg = TemplateGenerator()
		tg.generate_sizes(w_max=3,h_max=2)
		templates = tg.generate_templates()
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
