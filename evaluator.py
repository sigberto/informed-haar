import os
import re
import cv2
import numpy as np
from detector import Detector

class Evaluator:

    def __init__(self, test_dir, classifier):
        self.min_overlap_area = 0.5
        self.test_dir = test_dir
        self.classifier = classifier
        self.img_paths = self.get_paths(os.path.join(test_dir, 'pos.lst'))
        self.img_paths.extend(self.get_paths(os.path.join(test_dir, 'neg.lst')))
        annotation_paths = self.get_paths(os.path.join(test_dir, 'annotations.lst'))
        self.ground_truths = {} # dictionary of image path from test_dir to array of bounding boxes for that image, if any
        self.n_ground_truths = 0
        self.get_annotations(annotation_paths)
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
                lines = f.readlines()
            gtruths = [gtruth for gtruth in lines if 'Bounding box' in gtruth]
            matches = [re.search(r'\(.+\).*\(.+\).*\((\d+) ?, ?(\d+)\).*\((\d+) ?, ?(\d+)\).*', s) for s in gtruths]
            result = np.zeros((len(matches), 4))
            for idx, match in enumerate(matches):
                x = int(match.group(1))
                y = int(match.group(2))
                x2 = int(match.group(3))
                y2 = int(match.group(4))
                w = x2 - x
                h = y2 - y
                result[idx, :] = np.asarray([y, x, h, w])
            self.ground_truths[path] = result
            self.n_ground_truths += len(matches)

    def compare(self, bboxes, gtruths):
        bboxes = bboxes[bboxes[:, 0].argsort()[::-1]]  # sort bboxes by descending order of score
        if gtruths is None:
            return len(bboxes), 0
        n_matches = 0
        matched_gtruth = np.zeros((len(gtruths)))
        matched_bbox = np.zeros((len(bboxes)))
        for bb_idx, bbox in enumerate(bboxes):
            bbox = bbox[1:]  # ignore score
            highest_overlap = 0
            highest_overlap_pair = None
            for gt_idx, gtruth in enumerate(gtruths):
                if not matched_gtruth[gt_idx] and not matched_bbox[bb_idx]:
                    a = min(bbox[1] + bbox[3], gtruth[1] + gtruth[3])
                    b = max(bbox[1], gtruth[1])
                    dx = min(bbox[1] + bbox[3], gtruth[1] + gtruth[3]) - max(bbox[1], gtruth[1])
                    dy = min(bbox[0] + bbox[2], gtruth[0] + gtruth[2]) - max(bbox[0], gtruth[0])
                    if dx > 0 and dy > 0:
                        intn = dx*dy
                        union = bbox[2]*bbox[3] + gtruth[2]*gtruth[3] - intn
                        overlap = intn/union
                        if overlap > highest_overlap:
                            highest_overlap = overlap
                            highest_overlap_pair = (gt_idx, bb_idx)
            if highest_overlap > 0.5:
                n_matches += 1
                matched_gtruth[highest_overlap_pair[0]] = 1
                matched_bbox[highest_overlap_pair[1]] = 1
        n_misses = len(gtruths) - n_matches
        n_FP = len(bboxes) - n_matches
        return n_FP, n_misses

    def evaluate(self):
        detector = Detector(self.classifier)
        for img_path in self.img_paths:
            img = cv2.imread(img_path)
            bboxes = detector.detect_pedestrians(img)
            n_FP, n_misses = self.compare(bboxes, self.ground_truths.get(img_path, None))
            self.n_FP += n_FP
            self.n_misses += n_misses
        self.FPPI = 1.0*self.n_FP/len(self.img_paths) # rate of false positives per image
        self.miss_rate = 1.0*self.n_misses/self.n_ground_truths
        return self.FPPI, self.miss_rate


