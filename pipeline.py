import numpy as np
import pickle
from template_generator import TemplateGenerator
from ChannelFeatures import ChannelFeatures
from feature_generator import FeatureGenerator
from classifier import Classifier
import cv2
from os import path

class Pipeline():
	""" 
		Class for running the end-to-end Pipeline:

		1) Template generation
		2) ICF image feature extraction
		3) Final feature vector formulation
		4) Classifier training
		5) Top feature selection

	"""

	def __init__():
		""" Instantiates Pipeline's TemplateGenerator and ChannelFeatures """

		# =====[ Instantiate  ]=====
		self.tg = TemplateGenerator()
		self.cf = ChannelFeatures()
		

	def extract_features(self, dir_info=('train_us', 'pos.lst', 'neg.lst'), file_name=None):
		""" 
			Extracts features from directory provided

			Input: dir_info, file_name

			- dir_info: (tuple) with 3 strings: path to training images, list of positive image paths, list of negative image paths
			- file_name: string of name of the file to save X, Y to. If none given, then X and Y are not saved. 

			Output: X, Y

			- X: Input matrix (M training examples x N features) 
			- Y: Labels corresponding to M training examples

		"""

		#=====[ Use TemplateGenerator() to generate templates ]=====
		self.tg.generate_sizes()
		self.templates = tg.generate_templates()

		#=====[ Instantiate FeatureGenerator ]=====
		self.fg = FeatureGenerator(self.templates)

		pos_images, neg_images = self._get_image_paths(dir_info[0],dir_info[1],dir_info[2])

		#=====[ Create input matrix ]=====
		X = np.zeros((len(pos_images) + len(neg_images), len(templates) * cf.N_CHANNELS))
		X = self._get_feature_matrix(X, pos_images, 0)
		X = self._get_feature_matrix(X, neg_images, len(pos_images))
		print 'Obtained feature matrix with shape {}'.format(str(X.shape))

		#=====[ Create labels ]=====
		Y = self._make_labels(len(pos_image),len(neg_images))

		#=====[ If user specified a file name to save X, and Y to, pickle objects ]=====
		if file_name:
			pickle.dump({'input':X,'labels':Y}, open(file_name, 'wb'))

		return X, Y 

	def select_top_features(X, Y, num_features=None, num_estimators=100, max_depth=2, model_name=None):
		"""
			Trains boosted trees in order to calculate feature importance and select top num_features

			1) Instantiate classifier
			2) Train classifier 
			3) Save classifier
			4) Visualize feature weights
			5) Return indices of features with highest importance

			Input: X, Y, num_features, num_estimators, max_depth, model_name

			- X: Input matrix (M training examples x N features) 
			- Y: Labels corresponding to M training examples

			- num_features: number of indices to return for the top features. if no number specified, returns entire list of indices
			- num_estimators: number of estimators to use for training 
			- maximum depth for each of the estimators
			- model_name: string of name of the file to pickle the model to

			Output: list of num_features indices for the most important features 

		"""

		self.clf = Classifier(num_estimators, max_depth)
		self.clf.train(X,Y)

		#=====[ If user specified a model name to save clf, pickle object ]=====
		if model_name:
			pickle.dump(self.clf,open(model_name,'wb'))

		#=====[ Plot feature weights ]=====
		clf.plot_ft_weights('feature_weights.png')

		top_indices = clf.top_ft_indices(num_features)

		return top_indices


	def _get_feature_matrix(self, X, images, offset=0):
		""" Append feature vectors for each training example in images to X """

		# =====[ Iterate through images and calculate feature vector for each ]=====
		for idx, img in enumerate(images):

			cfeats = cf.compute_channels(cv2.imread(img))
			feature_vec = fg.generate_features(cfeats)

			# Add feature vector to input matrix
			X[idx + offset, :] = feature_vec


	# {pos,neg}_filename is a text file with file in the base_dir. They are either 'test_us/...' or 'train_us/...'
	def _get_image_paths(self, base_dir, pos_filename, neg_filename):
		""" Get list of image paths in base_dir from each file_name """

		with open(path.join(base_dir, pos_filename)) as f:
			pos_list = f.readlines()
			pos_list = [x.strip() for x in pos_list]
		with open(path.join(base_dir, neg_filename)) as f:
			neg_list = f.readlines()
			neg_list = [x.strip() for x in neg_list]

		print 'Loaded {} positive image paths and {} negative image paths'.format(str(len(pos_list)), str(len(neg_list)))
		return pos_list, neg_list

	def _make_labels(self, n_pos, n_neg):
		""" Takes number of positive and negative images and returns appropriate label vector """

		Y = np.zeros((n_pos + n_neg))
		Y[:len(n_pos)] = 1

		return Y 

