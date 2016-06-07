from classifier import Classifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from ChannelFeatures import ChannelFeatures
from feature_gen import FeatureGenerator
from template_generator import TemplateGenerator
import cv2

def _predict(clf, img_paths, templates): 
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

		if idx % 500 == 0:
			print '-----> Processing Image ', idx + 1
	
	print '-----> Processed all feature vectors'		

	#=====[ predict class for each feature_vector ]=====
	ys = clf.clf.predict(feature_vectors)

	return ys



def _formulate_stats(n_pos, n_neg, pos, neg):

	#=====[ Calculate False/True Pos/Neg ]=====
	FN= n_pos*(1-pos)
	TP = n_pos - FN
	accuracy = pos*n_pos/(n_pos+n_neg)

	FP = n_neg*(1-neg)
	TN = n_neg - FP
	accuracy += neg*n_neg/(n_pos + n_neg)

	#=====[ Calculate precision, recall, and f1 ]=====
	precision = float(TP)/(TP+FP)
	recall = float(TP)/(TP+FN)

	f1 = 2*precision*recall/(precision+recall)

	return (accuracy, f1)

def test(clf='classifiers/top_ft_classifier_100_200', templates=None):
	""" 
		Test accuracy against provided image paths and Y.
	"""

	clf = Classifier(clf=clf)

	tg = TemplateGenerator()
	tg.generate_sizes()
	
	templates = pickle.load(open(templates,'rb')) if templates else tg.generate_templates()
	templates = templates[:100]

	#=====[ Get positive and negative image paths ]=====
	raw_img_files = open('INRIAPerson/test_us/pos.lst','rb')
	pos_img_paths = ['INRIAPerson/test_us/pos/'+ path.strip() for path in raw_img_files.readlines()]
	raw_img_files.close()

	raw_img_files = open('INRIAPerson/test_us/neg.lst','rb')
	neg_img_paths = ['INRIAPerson/test_us/neg/'+ path.strip() for path in raw_img_files.readlines()]
	raw_img_files.close()

	#=====[ Get num of positive/negative images ]=====
	n_pos = len(pos_img_paths)
	n_neg = len(neg_img_paths)

	print '-----> Testing on ', str(n_pos + n_neg), ' images'

	#=====[ Get positive image accuracy ]=====
	Y_pos = np.ones((len(pos_img_paths)))
	pos_accuracy = accuracy_score(Y_pos, _predict(clf, pos_img_paths, templates))

	print '-----> Positive image accuracy: ', pos_accuracy

	#=====[ Get negative image accuracy ]=====
	Y_neg = np.zeros((len(neg_img_paths)))
	neg_accuracy = accuracy_score(Y_neg, _predict(clf, neg_img_paths, templates))

	print '-----> Negative image accuracy: ', neg_accuracy

	accuracy, f1 = _formulate_stats(n_pos, n_neg, pos_accuracy, neg_accuracy)
	
	print 'Accuracy: ', accuracy
	print 'F1 Score: ', f1




