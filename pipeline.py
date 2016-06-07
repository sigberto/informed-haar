from os import path
import numpy as np
import pickle
import cv2
import sys
import imutils

sys.path.append('classifiers')
sys.path.append('modules')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from template_generator import TemplateGenerator
from ChannelFeatures import ChannelFeatures
from feature_generator import FeatureGenerator
from classifier import Classifier
from linear_detector import Detector as LinearScaleDetector
from detector import Detector
from evaluator import Evaluator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import test_classification


class Pipeline:
	"""
		Class for running the end-to-end Pipeline:

		1) Template generation
		2) ICF image feature extraction
		3) Final feature vector formulation
		4) Classifier training
		5) Pedestrian Detection
		6) Evaluation

	"""

	def __init__(self):
		""" Instantiates Pipeline's TemplateGenerator and ChannelFeatures """

		# =====[ Instantiate  ]=====
		self.tg = TemplateGenerator()
		self.cf = ChannelFeatures()
		self.detector = None

	def extract_features(self, num_ft=100, dir_info=('INRIAPerson/train_us', 'pos.lst', 'neg.lst'), template_file='processed_data/top_templates_1000.p', file_name=None):
		"""
			Extracts features from directory provided

			Input: num_ft, dir_info, file_name

			- num_ft: Top N features to be used for classification
			- dir_info: (tuple) with 3 strings: path to training images, list of positive image paths, list of negative image paths
			- file_name: string of name of the file to save X, Y to. If none given, then X and Y are not saved.

			Output: X, Y

			- X: Input matrix (M training examples x N features)
			- Y: Labels corresponding to M training examples
		"""

		# =====[ Use TemplateGenerator() to generate templates ]=====
		self.templates = pickle.load(open(template_file,'r'))
		self.templates = self.templates[:num_ft]

		# =====[ Instantiate FeatureGenerator ]=====
		self.fg = FeatureGenerator(self.templates)

		pos_images, neg_images = self._get_image_paths(dir_info[0], dir_info[1], dir_info[2])

		# =====[ Create input matrix ]=====
		print '-----> Total images to process: ', len(pos_images) + len(neg_images)
		X = np.zeros((len(pos_images) + len(neg_images), len(self.templates)))
		X = self._get_feature_matrix(X, pos_images, 0)
		X = self._get_feature_matrix(X, neg_images, len(pos_images) - 1)
		print '-----> Obtained feature matrix with shape {}'.format(str(X.shape))

		# =====[ Create labels ]=====
		Y = self._make_labels(len(pos_images), len(neg_images))

		# =====[ If user specified a file name to save X, and Y to, pickle objects ]=====
		if file_name:
			pickle.dump({'input': X, 'labels': Y}, open(file_name, 'wb'))
			print '-----> Successfully formulated and saved X and Y'

		return X, Y

	def train_classifier(self, X, Y, num_estimators=200, max_depth=2, save_to_file=None):
		"""
			Trains boosted trees in order to calculate feature importance and select top num_features
	
			1) Instantiate classifier
			2) Train classifier
			3) Save classifier
			4) Visualize feature weights
	
			Input: X, Y, num_features, num_estimators, max_depth, save_to_file
	
			- X: Input matrix (M training examples x N features)
			- Y: Labels corresponding to M training examples
	
			- num_features: number of indices to return for the top features. if no number specified, returns entire list of indices
			- num_estimators: number of estimators to use for training
			- maximum depth for each of the estimators
			- save_to_file: string of name of the file to pickle the model to
	
			Output: list of num_features indices for the most important features
	
		"""
	
		# self.clf = Classifier(num_estimators, max_depth)

		# print '-----> Training'
		# self.clf.train(X, Y)
		# print '-----> Training Complete'
	
		# #=====[ If user specified a model name to save clf, pickle object ]=====
		# if save_to_file:
		# 	pickle.dump(self.clf, open(save_to_file, 'wb'))
		# 	print '-----> Saved classifier as ', save_to_file

		self.clf = pickle.load(open(save_to_file,'r'))
	
		#=====[ Plot feature weights ]=====
		self._plot_ft_weights('feature_weights.png')


	def test_classifier(self, model_name, template_file='processed_data/top_templates_1000.p'):
		
		test_classification.test(model_name, template_file)


	def detect(self, img_path=None, output_file_prefix='', num_ft=100, offset=0, scaling_factor = 1.2, scaling_iters=3, nms=0.5, clf=None, templates=None, linear_scaling=False):
		
		#=====[ Load our classifier and templates ]=====
		clf = pickle.load(open(clf)) if clf else pickle.load(open('classifiers/top_ft_classifier_100_200', 'r'))
		templates = pickle.load(open(templates)) if templates else pickle.load(open('processed_data/top_templates_1000.p','r'))
	
		#=====[ Get top templates ]=====
		templates = templates[:num_ft]

		#=====[ Instantiate our feature generator ]=====
		fg = FeatureGenerator(templates)

		#=====[ Instantiate our detector ]=====
		if linear_scaling:
			self.detector = LinearScaleDetector(clf.clf, fg,scaling_factor=scaling_factor,scaling_iters=scaling_iters, nms=nms)
		else:
			self.detector = Detector(clf.clf, fg,scaling_factor=scaling_factor,scaling_iters=scaling_iters, nms=nms)

		#=====[ If a specific image path is given, then we do not evaluate, just detect the pedestrian and draw bounding boxes ]=====
		if img_path:
			_, bbs = self.detector.detect_pedestrians(img_path)
			self._draw_bbs(img_path, bbs)
		
		else:
			#=====[ Instantiate our evaluator and evaluate ]=====
			evaluator = Evaluator('INRIAPerson/Test', self.detector)
			FPPI, miss_rate = evaluator.evaluate(output_file_prefix,offset)

			print '\nFPPI: {}\nMiss rate: {}\n'.format(FPPI, miss_rate)

	def get_stats(self, output_file_prefix='', num_images=100):
		""" Reports number of people, matches, misses, and false positives per experiment from generated output files """

		#=====[ Instantiate our evaluator ]=====
		evaluator = Evaluator('INRIAPerson/Test')

		num_people, num_hits, num_misses, num_FP, num_processed = evaluator.aggregate_stats(output_file_prefix, num_images)

		#=====[ Print statistics ]=====
		print '-----> Stats for ' + output_file_prefix + ' ( ' + str(num_processed) + '/' + str(num_images) +' processed) :\n\n'
		print 'Miss Rate: ' + str(float(num_misses)/num_people)
		print 'False Positives: ' + str(num_FP)
		print 'FPPI: ' + str(float(num_FP)/num_images)
		print 'Hits: ' + str(num_hits) 
		print 'Misses: ' + str(num_misses) 
		print 'Total People: ' + str(num_people) 


	def _get_feature_matrix(self, X, images, offset=0):
		""" Append feature vectors for each training example in images to X """

		# =====[ Iterate through images and calculate feature vector for each ]=====
		for idx, img in enumerate(images):

			try:
				cfeats = self.cf.compute_channels(cv2.imread(img))
				feature_vec = self.fg.generate_features(cfeats)

				#=====[ Add feature vector to input matrix ]=====
				X[idx + offset, :] = feature_vec

			except Exception as e:
				print 'Could not add image at index: ', idx + offset

		return X

	def _get_image_paths(self, base_dir, pos_filename, neg_filename):
		""" Get list of image paths in base_dir from each file_name """

		with open(path.join(base_dir, pos_filename)) as f:
			pos_list = f.readlines()
			pos_list = [base_dir + '/pos/' + x.strip() for x in pos_list]
		with open(path.join(base_dir, neg_filename)) as f:
			neg_list = f.readlines()
			neg_list = [base_dir + '/neg/' + x.strip() for x in neg_list]

		print '-----> Loaded {} positive image paths and {} negative image paths'.format(str(len(pos_list)),
																				  str(len(neg_list)))
		return pos_list, neg_list

	def _make_labels(self, n_pos, n_neg):
		""" Takes number of positive and negative images and returns appropriate label vector """

		Y = np.zeros((n_pos + n_neg))
		Y[:n_pos] = 1

		return Y

	def _plot_ft_weights(self, file_name):
		""" Generates heatmap of activated cells within our pedestrian model """

		base_clf = pickle.load(open('classifiers/full_feature_2000_est_classifier'))
		num_ft_base = len(base_clf.clf.feature_importances_)
		num_ft_cur = len(self.clf.clf.feature_importances_)

		feats = [(base_clf.clf.feature_importances_, num_ft_base), (self.clf.clf.feature_importances_, num_ft_cur)]

		#=====[ Get feature information corresponding to each template ]=====
		feature_info = pickle.load(open('processed_data/feature_info.p', 'r'))

		w_viz = np.zeros((20, 20))


		for off, impt in enumerate(feats):


			feat_impt = impt[0]

			#=====[ Sum weights from each feature corresponding to cells in our templates ]=====
			for idx, w in enumerate(feat_impt):
				x, y, size, k = feature_info[idx]
				w, h = size
				w_viz[y:y + h, 10*off+x:10*off+x + w] += w

			#=====[ Normalize weight visualization matrix and display ]=====
			w_viz[:,10*off:10*(off+1)] = w_viz[:,10*off:10*(off+1)] / np.max(w_viz[:,10*off:10*(off+1)])
		
		im = plt.matshow(w_viz, cmap='Reds')
			
		#=====[ Formulate title ]=====
		title = 'Cell Activations\n\n' + str(num_ft_base) + ' Features     |     ' + str(num_ft_cur) + ' Features\n'
		
		plt.title(title)

		ax = plt.gca()

		#=====[ create an axes on the right side of ax. The width of cax will be 5% 
		#=====[ of ax and the padding between cax and ax will be fixed at 0.05 inch.
		divider = make_axes_locatable(ax)
		cax = divider.append_axes("right", size="5%", pad=0.05)
		f = plt.gcf()
		f.set_size_inches(20,5)
		plt.colorbar(im, cax=cax)

		plt.savefig(file_name)

	def _draw_bbs(self, img_path, bbs):
		""" Draws bounding boxes on specified image and displays it """
					
		img = cv2.imread(img_path)
		# scale = img.shape[0]/300.0
		
		# img = imutils.resize(img,height=300)
		scale = 1

		for box in bbs:
			cv2.rectangle(img,(int(box[2]*scale),int(box[1]*scale)),(int((box[2]+box[4])*scale),int((box[1]+box[3])*scale)),(0,255,0),2)    

		cv2.imwrite('detected_'+img_path,img)

		to_show = mpimg.imread('detected_'+img_path)
		plt.imshow(to_show)


