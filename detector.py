import numpy as np
import cv2

from ChannelFeatures import ChannelFeatures
from feature_generator import FeatureGenerator


class Detector():

	def __init__(self, weight_indices, clf, window_size=(120,60), scaling_factor=1.09, scaling_iters=10, window_step=6):

		self.indices = weight_indices
		self.clf = clf
		self.window_size = window_size
		self.window_step = window_step

		self.scaling_factor = scaling_factor
		self.scaling_iters = scaling_iters

		self.cf = ChannelFeatures()


	def detect_pedestrians(self, img_path):
		"""
			Detects pedestrians in an image.

			1) Slides bounding box window over the image
			2) Computes detection score using weights from boosted tree classifier
			3) Keeps the bounding box if the score is above a certain threshold
			4) Runs non-maximal suppression (NMS)

			Input: img_path

			img_path: path to image file

			Output: list of bounding boxes and scores 

		"""

		img = cv2.imread(img_path)
		height, width, channels = img.shape
		win_h, win_w = self.size

		for it_num in range(self.scaling_iters):

			if it_num > 0:
						pass
						#=====[ Appropriately scale img ]=====
						#img = 
						#=====[ TO DO ]=====

			cfeats = cf.compute_channels(img)

			for y in range(height/self.window_step - win_h):
				for x in range(width/self.window_step - win_w):

					feature_vec = np.asarray(fg.generate_features(cfeats[y*win_h:(y+1)*win_h,x*win_w:(x+1)*win_w])[self.indices])


def _get_bounding_boxes(img_path):
	""" 
		Returns 2D array of bounding boxes (M bounding boxes x 5 characteristics per bounding box)
	"""
        
        img = cv2.imread(img_path)

        oheight, owidth, channels = img.shape
        win_h, win_w = self.window_size

        #=====[ Collect bounding boxes for each scaling iteration ]=====
        for it_num in range(self.scaling_iters):

        	#=====[ Scale image if not on first iteration ]=====
            if it_num > 0:
                img = cv2.resize(img,(it_num*self.scaling_factor*owidth, it_num*self.scaling_factor*oheight))
    
            height, width, _ = img.shape
        
            y_range = (height - win_h)/self.window_step + 1
            x_range = (width - win_w)/self.window_step + 1

            cfeats = cf.compute_channels(img)

            #=====[ Slide window across entirety of image and calculate bounding box at each step ]=====
            for y in range(y_range):
                for x in range(x_range):
                    
                    y_pix = y*self.window_step
                    x_pix = x*self.window_step
                    
                    #=====[ Score the bounding box ]=====
                    feature_vec = np.asarray(self.fg.generate_features(cfeats[y:]))
                    score = self.clf.predict(feature_vec)
                    
                    #=====[ Scale and store bounding box ]=====
                    scale = self.scaling_factor*it_num if it_num else 1
                    bounding_boxes.append([score, y_pix/scale, x_pix/scale, win_h/scale, win_w/scale])


            return np.matrix(bounding_boxes)


