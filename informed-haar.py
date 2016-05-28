import cv2
from ChannelFeatures import ChannelFeatures


img = cv2.imread('INRIAPerson/Train/pos/crop001001.png')
chnft = ChannelFeatures(img)
features = chnft.compute_channels()
print features.shape