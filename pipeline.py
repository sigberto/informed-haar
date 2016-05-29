import numpy as np
import pickle
from template_generator import TemplateGenerator
from ChannelFeatures import ChannelFeatures
from feature_generator import FeatureGenerator
import cv2
from os import path

def get_feature_matrix(X, images, offset)

	#=====[ Iterate through images and calculate feature vector for each ]=====
	for idx, img in enumerate(images):
    
   		cfeats = cf.compute_channels(cv2.imread(img))
    		feature_vec = fg.generate_features(cfeats)

    		# Add feature vector to input matrix
    		X[idx + offset,:] = feature_vec

def get_image_paths(base_dir, pos_filename, neg_filename)
    with open(path.join(base_dir, pos_filename)) as f:
        pos_list = f.readlines()
        pos_list = [x.strip() for x in pos_list]
    with open(path.join(base_dir, neg_filename)) as f:
        neg_list = f.readlines()
        neg_list = [x.strip() for x in neg_list]
    return pos_list, neg_list

#=====[ Generate templates ]=====
tg = TemplateGenerator()
tg.generate_sizes()
templates = tg.generate_templates()

print 'Created %d templates' % (len(templates))

#=====[ Instantiate feature and channel feature generators ]=====
fg = FeatureGenerator(templates)
cf = channelFeatures()

pos_images, neg_images = get_image_paths('../train_us/', 'pos.lst', 'neg.lst')

# Instantiate X 
X = np.zeros((len(pos_images)+ len(neg_images),len(images)*len(templates)*cf.N_CHANNELS))
Y = []

X = get_feature_matrix(X, pos_images, 0)
X = get_feature_matrix(X, neg_images, len(pos_images))


pickle.dump(X,open('features.p','wb'))

