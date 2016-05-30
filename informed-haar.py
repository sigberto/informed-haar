import cv2
from ChannelFeatures import ChannelFeatures
# from ImagePreprocessor import ImagePreprocessor
#
# ip = ImagePreprocessor()
# ip.preprocess_images()
#
img = cv2.imread('INRIAPerson/Test/pos/person_and_bike_058.png')
chnft = ChannelFeatures()
features = chnft.compute_channels(img, resize=False)
print features.shape
# print features.shape

# from nms import non_max_suppression
# import numpy as np
#
#
# dets = np.asarray([
#     #[score, y, x, h, w]
#     [0.56, 60, 60, 30, 30],
#     [0.67, 50, 50, 30, 30],
#     [0.61, 55, 55, 30, 30],
#     [0.78, 120, 120, 30, 30],
# ])
#
# picks = non_max_suppression(dets, 0.3)
# print picks