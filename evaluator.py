import os
import cv2

class Evaluator:

    def __init__(self, test_dir):
        self.min_score = 0.5
        self.test_dir = test_dir
        self.ground_truths = {} # will be dictionary of image path from test_dir to array of bounding boxes for that image, if any
        self.img_paths = self.get_paths(os.path.join(test_dir, 'pos.lst'))
        self.img_paths.extend(self.get_paths(os.path.join(test_dir, 'neg.lst')))
        anotation_paths = self.get_paths('anotations.lst')
        self.anotations = self.get_anotations(anotation_paths)
        self.n_FP = 0
        self.FPPI = None
        self.n_misses = 0
        self.miss_rate = None

    def get_paths(self, list_file_path):

    def get_anotations(self, anotation_paths):

    def evaluate(self):
        detector = Detector()
        for img_path in self.img_paths:
            img = cv2.imread(img_path)
            bboxes = detector.detect(img)
            n_FP, n_misses = self.compare(bboxes, self.ground_truths.get(img_path, None))
            self.n_FP += n_FP
            self.n_misses += n_misses
        self.FPPI = 1.0*self.n_FP/len(self.img_paths) # rate of false positives per image
        self.miss_rate = 1.0*self.n_misses/len(self.img_paths)
        return self.FPPI, self.miss_rate


