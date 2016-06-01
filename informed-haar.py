import cv2
from ChannelFeatures import ChannelFeatures
# from ImagePreprocessor import ImagePreprocessor
#
# ip = ImagePreprocessor()
# ip.preprocess_images()
#
# img = cv2.imread('INRIAPerson/Test/pos/person_and_bike_058.png')
# chnft = ChannelFeatures()
# features = chnft.compute_channels(img, resize=False)
# print features.shape
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

from evaluator import Evaluator
import numpy as np
# import re
# s = 'Bounding box for object 2 "PASperson" (Xmin, Ymin) - (Xmax, Ymax) : (229, 221) - (381, 793)'
# matches = re.search(r'\(.+\).*\(.+\).*\((\d+) ?, ?(\d+)\).*\((\d+) ?, ?(\d+)\).*', s)
# print matches.group(1)
# print matches.group(2)
# print matches.group(3)
# print matches.group(4)

bboxes = np.asarray([
    #[score, y, x, h, w]
    [0.9, 5, 0, 30, 30],
])

print bboxes[bboxes[:, 0].argsort()[::-1]]

gtruths = np.asarray([
    [5, 0, 30, 30],
    [0, 0, 30, 30],
])


evaluator = Evaluator('INRIAPerson/Test', None)
FPPI, miss_rate = evaluator.compare(bboxes, gtruths)
print FPPI, miss_rate