import numpy as np
import cv2
import imutils

from ChannelFeatures import ChannelFeatures
from feature_generator import FeatureGenerator
import nms


class Detector:
    """ 
        The Detector class is used to detect pedestrians in images by locating bounding boxes with high probabilities
        containing a pedestrian 
    """

    def __init__(self, clf, fg, window_size=(120,60), scaling_factor=1.2, scaling_iters=3, window_step=6):
        """ Instantiates the detector class:
        
            Input: weight_indices, weights, window_size, scaling_factor, scaling_iters, window_step
            
            - weight_indices: the indices of the features that will be used to score a window in the image
            - weights: the weights used to compute a score for a feature vector associated with a window in the image
            - fg: FeatureGenerator() object used to generate feature vectors for window in an image
            - window_size: the size of the sliding window defaults to 120 x 60
            - scaling_factor: factor by which we scale the image on each successive scaling iteration
            - scaling_iters: the number of times we scale the image
            - window_step: the amount of pixels stepped over on each slide of the window
        """

        self.clf = clf
        self.window_size = window_size
        self.window_step = window_step

        self.scaling_factor = scaling_factor
        self.scaling_iters = scaling_iters

        self.cf = ChannelFeatures()
        self.fg = fg

    def detect_pedestrians(self, img_path):
        """
            Detects pedestrians in an image.

            1) Slides bounding box window over the image
            2) Computes detection score using weights from boosted tree classifier
            3) Keeps the bounding box if the score is above a certain threshold
            4) Runs non-maximal suppression (NMS) on bounding boxes

            Input: img_path

            - img_path: path to image file

            Output: list of bounding boxes and scores 

        """
        
        candidate_bbs = self._get_bounding_boxes(img_path)
        bbs = None
        if len(candidate_bbs) > 1:
            bbs = nms.non_max_suppression(np.asarray(candidate_bbs), overlapThresh=0.5)
        elif len(candidate_bbs) == 0:
            bbs = candidate_bbs

        return candidate_bbs, bbs

    def _get_bounding_boxes(self, img_path, start_h=120, start_w=60):
        """ 
            Returns 2D array of bounding boxes (M bounding boxes x 5 characteristics per bounding box)
        """
        
        bounding_boxes = []
    
        
        img = cv2.imread(img_path)
        raw_height, raw_width, channels = img.shape

        if raw_height/start_h > raw_width/start_w:
            img = imutils.resize(img, width=min(start_w,img.shape[1]))
        else:
            img = imutils.resize(img, height=min(start_h,img.shape[0]))
            
        cv2.imwrite('resized_img.jpeg',img)

        oheight, owidth, channels = img.shape
        win_h, win_w = self.window_size

        count = 0
        
        #=====[ Collect bounding boxes for each scaling iteration ]=====
        for it_num in range(1, self.scaling_iters + 1):

            #=====[ Scale image if not on first iteration ]=====
            if it_num > 0:
                img = cv2.resize(img,(int(it_num*self.scaling_factor*owidth), int(it_num*self.scaling_factor*oheight)))

            height, width, _ = img.shape

            y_range = (height - win_h)/self.window_step + 1
            x_range = (width - win_w)/self.window_step + 1
            
            cfeats = self.cf.compute_channels(img)
            
            #=====[ Slide window across entirety of image and calculate bounding box at each step ]=====
            for y in range(y_range):
                for x in range(x_range):

                    y_pix = y*self.window_step
                    x_pix = x*self.window_step

                    #=====[ Score the bounding box ]=====
                    feature_vec = np.asarray(self.fg.generate_features(cfeats[y:y+win_h,x:x+win_w]))
                    
                    score = self.clf.predict_proba([feature_vec])[0,1]
		    #score = 1
                    #=====[ Scale and store bounding box ]=====
                    scale = self.scaling_factor*it_num if it_num else 1
		   
		    scale *= float(oheight)/raw_height
                    count += 1
                    
                    if score > 0.5:
                        bounding_boxes.append([score, int(y_pix/scale), int(x_pix/scale), int(win_h/scale), int(win_w/scale)])


            print 'Went through %d total candidate BBs' %(count)
        return np.matrix(bounding_boxes)
        
    def _calculate_total_iters(self, img):
        """ Calculates total number of bounding box scores to be calculated """
        
        oheight, owidth, channels = img.shape
        win_h, win_w = self.window_size
        
        iters = 0
        
        for it_num in range(self.scaling_iters):

            #=====[ Scale image if not on first iteration ]=====
            if it_num > 0:
                img = cv2.resize(img,(int(it_num*self.scaling_factor*owidth), int(it_num*self.scaling_factor*oheight)))

            height, width, _ = img.shape

            y_range = (height - win_h)/self.window_step + 1
            x_range = (width - win_w)/self.window_step + 1
            
            iters += y_range*x_range
        
        return iters
