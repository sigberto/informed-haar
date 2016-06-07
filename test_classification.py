from classifier import Classifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
from ChannelFeatures import ChannelFeatures
from feature_generator import FeatureGenerator
from template_generator import TemplateGenerator
import cv2

def predict(clf, img_paths, templates): 
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
	ys = clf.clf.predict(feature_vectors)

	return ys

def test(clf, img_paths, Y, templates):
	""" 
		Test accuracy against provided image paths and Y.
	"""
	ys = predict(clf, img_paths, templates)

	accuracy = accuracy_score(Y, ys)
	return accuracy








clf = Classifier(clf='BoostedTreeclassifier.p')
tg = TemplateGenerator()
tg.generate_sizes()
templates = tg.generate_templates()
#templates = pickle.load(open('top_templates_1000.p','rb'))
#templates = templates[:100]

#=====[ Get positive and negative image paths ]=====
raw_img_files = open('../test_us/pos.lst','rb')
pos_img_paths = ['../test_us/pos/'+ path.strip() for path in raw_img_files.readlines()]
raw_img_files.close()

raw_img_files = open('../test_us/neg.lst','rb')
neg_img_paths = ['../test_us/neg/'+ path.strip() for path in raw_img_files.readlines()]
raw_img_files.close()

print '-----> Aggregated image paths'

#=====[ Get num of positive/negative images ]=====
n_pos = len(pos_img_paths)
n_neg = len(neg_img_paths)

#=====[ Get positive image accuracy ]=====
Y_pos = np.ones((len(pos_img_paths)))
accuracy = test(clf, pos_img_paths, Y_pos, templates)

print '-----> Positive image accuracy: ', accuracy

FN= n_pos*(1-accuracy)
TP = n_pos - FN
accuracy = accuracy*n_pos/(n_pos+n_neg)

#=====[ Get negative image accuracy ]=====
Y_neg = np.zeros((len(neg_img_paths)))
neg_accuracy = test(clf, neg_img_paths, Y_neg, templates)

print '-----> Negative image accuracy: ', neg_accuracy

FP = n_neg*(1-neg_accuracy)
TN = n_neg - FP
accuracy += neg_accuracy*n_neg/(n_pos + n_neg)

precision = float(TP)/(TP+FP)
recall = float(TP)/(TP+FN)
print 'Accuracy on %d images: %f ' % (n_pos+n_neg, accuracy)
print 'F1 on %d images: %f' % (n_pos + n_neg, 2*precision*recall/(precision+recall))

