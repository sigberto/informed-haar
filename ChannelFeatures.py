import cv2
import numpy as np
import imutils
from math import trunc

class ChannelFeatures:

    CELL_SIZE = 6  # pixels
    IMG_HEIGHT = 120
    IMG_WIDTH = 60
    H_CELLS = trunc(IMG_HEIGHT/CELL_SIZE)
    W_CELLS = trunc(IMG_WIDTH/CELL_SIZE)
    N_CHANNELS = 10
    NUM_HOG_BINS = 6

    def __init__(self, img):
        img = imutils.resize(img, width=self.IMG_WIDTH, height=self.IMG_HEIGHT)
        img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
        self.img = cv2.GaussianBlur(img, ksize=(0, 0), sigmaX=.87)
        self.feats = np.zeros((self.H_CELLS, self.W_CELLS, self.N_CHANNELS))

    # @staticmethod
    # def _absmax(x):
    #     mx = np.amax(x, axis=(0, 1))
    #     mn = np.amin(x, axis=(0, 1))
    #     length = len(x[0, 0, :])
    #     maxvals = np.zeros((length,))
    #     for i in xrange(length):
    #         maxvals[i] = mx[i] if abs(mx[i]) > abs(mn[i]) else mn[i]
    #     return maxvals

    def _pool(self, vol):
        _, _, depth = vol.shape
        feats = np.zeros((self.H_CELLS, self.W_CELLS, depth))
        for i in xrange(self.H_CELLS-1):
            h_offset = i*self.CELL_SIZE
            for j in xrange(self.W_CELLS-1):
                w_offset = j*self.CELL_SIZE
                subluv = vol[h_offset:h_offset + self.CELL_SIZE, w_offset:w_offset + self.CELL_SIZE, :]
                feats[i, j, :] = np.sum(subluv)
        return feats

    def _compute_luv(self):
        luv = cv2.cvtColor(self.img, cv2.COLOR_BGR2LUV)
        feats = self._pool(luv)
        return feats

    def _compute_gradients(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        sobelx64f = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        abs_sobel64f = np.absolute(sobelx64f)
        sobel_x = np.uint8(abs_sobel64f)
        sobel_x = self._pool(sobel_x.reshape(H, W, 1))

        sobely64f = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        abs_sobel64f = np.absolute(sobely64f)
        sobel_y = np.uint8(abs_sobel64f)
        sobel_y = self._pool(sobel_y.reshape(H, W, 1))
        return np.dstack((sobel_x, sobel_y))

    def _reshape_hog_feats(self, linear_hog_feats):
        hog_feats = np.zeros((self.H_CELLS, self.W_CELLS, self.NUM_HOG_BINS))
        ind = 0
        for i in xrange(self.H_CELLS):
            for j in xrange(self.W_CELLS):
                for k in xrange(self.NUM_HOG_BINS):
                    hog_feats[i, j, k] = linear_hog_feats[ind]
                    ind += 1
        return hog_feats

    def _compute_hog(self):
        winSize = (self.IMG_WIDTH, self.IMG_HEIGHT)
        blockSize = (self.CELL_SIZE, self.CELL_SIZE)
        blockStride = (self.CELL_SIZE, self.CELL_SIZE)
        cellSize = (self.CELL_SIZE, self.CELL_SIZE)
        nbins = self.NUM_HOG_BINS
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma,
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        hog_feats = hog.compute(self.img)
        return hog_feats.reshape(self.H_CELLS, self.W_CELLS, self.NUM_HOG_BINS)

    def compute_channels(self):
        # LUV channels
        luv = self._compute_luv()

        # Gradient magnitudes in X and Y (Sobel filters)
        grads = self._compute_gradients()

        # Gradient histogram channels (HOG, 6 bins)
        hog = self._compute_hog()

        channels = np.dstack((luv, grads, hog))
        return channels



