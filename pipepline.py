import numpy as np
import pickle
from template_generator import TemplateGenerator
from ChannelFeatures import ChannelFeatures
from feature_generator import FeatureGenerator
import cv2

#=====[ Generate templates ]=====
tg = TemplateGenerator()
tg.generate_sizes()
templates = tg.generate_templates()

print 'Created %d templates' % (len(templates))

#=====[ Instantiate feature and channel feature generators ]=====
fg = FeatureGenerator(templates)
cf = channelFeatures()

# Instantiate X 
X = np.zeros((len(images),len(images)*len(templates)*cf.N_CHANNELS))
Y = []

#=====[ Iterate through images and calculate feature vector for each ]=====
for idx, img in enumerate(images):
    
    cfeats = cf.compute_channels(cv2.imread(img))
    feature_vec = fg.generate_features(cfeats)

    # Add feature vector to input matrix
    X[idx,:] = feature_vec
    Y.append(label)

Y = np.array(Y)

pickle.dumps(open('features.p','wb'))