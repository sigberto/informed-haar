import os
import cv2
from detector import Detector

class Evaluator:

    def __init__(self, test_dir, classifier):
        self.min_overlap_area = 0.5
        self.test_dir = test_dir
        self.classifier = classifier
        self.img_paths = self.get_paths(os.path.join(test_dir, 'pos.lst'))
        self.img_paths.extend(self.get_paths(os.path.join(test_dir, 'neg.lst')))
        annotation_paths = self.get_paths(os.path.join(test_dir, 'annotations.lst'))
        self.ground_truths = self.get_annotations(annotation_paths)  # dictionary of image path from test_dir to array of bounding boxes for that image, if any
        self.n_FP = 0
        self.FPPI = None
        self.n_misses = 0
        self.miss_rate = None

    def get_paths(self, list_file_path):
        with open(list_file_path) as f:
            paths = f.readlines()
            paths = ['INRIAPerson/' + x.strip() for x in paths]
        return paths

    def get_annotations(self, annotation_paths):
        for path in annotation_paths:

            with open(path) as f:


    def compare(self, bboxes, gtruths):
        pass

    def evaluate(self):
        detector = Detector(self.classifier)
        for img_path in self.img_paths:
            img = cv2.imread(img_path)
            bboxes = detector.detect_pedestrians(img)
            n_FP, n_misses = self.compare(bboxes, self.ground_truths.get(img_path, None))
            self.n_FP += n_FP
            self.n_misses += n_misses
        self.FPPI = 1.0*self.n_FP/len(self.img_paths) # rate of false positives per image
        self.miss_rate = 1.0*self.n_misses/len(self.img_paths)
        return self.FPPI, self.miss_rate


