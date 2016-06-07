import sys
import os
import re
import cv2
import numpy as np
from detector import Detector

class Evaluator:

    def __init__(self, test_dir, detector=None):
        self.min_overlap_area = 0.5
        self.test_dir = test_dir
        self.detector = detector
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
            img_path = [line for line in lines if 'filename' in line]
            img_path = 'INRIAPerson/' + img_path[0].split('"')[1]
            self.ground_truths[img_path] = result
            self.n_ground_truths += len(matches)

    def compare(self, bboxes, gtruths):
        if bboxes is None and gtruths is None:
            return 0, 0
        if bboxes is None:
            return 0, len(gtruths)
        if gtruths is None:
            return len(bboxes), 0
        bboxes = bboxes[bboxes[:, 0].argsort()[::-1]]  # sort bboxes by descending order of score
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
        return (n_FP, n_misses, n_matches)

    def save_image_results(self, img_path, bboxes, gtruths, n_FP, n_misses, prefix='', offset=0):
        if gtruths is None:
            gtruths_str = ''
        else:
            gtruths_str = list(gtruths)
        if bboxes is None:
            bboxes_str = ''
        else:
            bboxes_str = list(bboxes)
        txt_file_path = img_path.replace('png', 'txt')
        txt_file_path = txt_file_path.replace('jpg', 'txt')

        #=====[ Hard coded index for image path start ]=====
        index = 21
        txt_file_path = txt_file_path[:index] + prefix + txt_file_path[index:]
        print 'Saving image at: ', txt_file_path
        with open(txt_file_path, 'wb') as f:
            try:
                line = 'bboxes: ' + ', '.join(['; '.join([str(y) for y in x]) for x in bboxes_str]) + \
                            '\ngtruths: ' + ' '.join(['; '.join([str(y) for y in x]) for x in gtruths_str]) + \
                            '\nn_FP: ' + str(n_FP) + \
                            '\nn_misses: ' + str(n_misses)
                f.write(line)
            except Exception as e:
                line = 'did not work'
                f.write(line)
                print e
                #   print 'bboxes: ' + str(bboxes_str) 
    

    def evaluate(self, path_prefix='', offset=0):
        matches = []
        for idx, img_path in enumerate(self.img_paths[offset:]):
            try:
		    print '-----> Processing img: ',offset +idx + 1,' path:',img_path
		    _, bboxes = self.detector.detect_pedestrians(img_path)
		    n_FP, n_misses, n_matches = self.compare(bboxes, self.ground_truths.get(img_path, None))
		    matches.append(n_matches)
		    print 'n_Match: ', n_matches
		    print 'n_FP: ', n_FP
		    print 'n_misses: ', n_misses
		    self.save_image_results(img_path, bboxes, self.ground_truths.get(img_path, None), n_FP, n_misses, prefix=path_prefix, offset=offset)
		    self.n_FP += n_FP
		    self.n_misses += n_misses
		    print 'Finished processing img: %d/%d' % (idx+offset+1, len(self.img_paths))
            except KeyboardInterrupt:
		    print 'FPPI: ', self.n_FP
		    print 'Misses: ', self.n_misses
		    print 'Total Matches: ', sum(matches)
		    print '%d/%d images processed' % (idx+offset+1, len(self.img_paths))
		    print 'Matches: ', matches
		    sys.exit()
            except Exception as e:
		    print e
		    print 'Could not Process image: ', offset +idx + 1 
        self.FPPI = 1.0*self.n_FP/len(self.img_paths) # rate of false positives per image
        self.miss_rate = 1.0*self.n_misses/self.n_ground_truths
        print 'FPPI: ',self.FPPI
        print 'Miss Rate: ', self.miss_rate
        return self.FPPI, self.miss_rate



    def aggregate_stats(self, path_prefix='', num_images=None):
        """ Aggregates stats for an experiment for num_images """

        #=====[ If num_images not specified, set to length of the number of paths ]=====
        num_images = num_images if num_images else len(self.img_paths)

        num_processed = 0
    	num_people = 0
    	num_FP = 0
    	num_misses = 0

        for idx, img_path in enumerate(self.img_paths[:num_images]):
	    try:
		    added = False

		    #=====[ Formulate new image path based on prefix ]=====
		    txt_file_path = img_path.replace('png', 'txt')
		    txt_file_path = txt_file_path.replace('jpg', 'txt')
		    index = 21 # Hard coded index for directory to image paths
		    txt_file_path = txt_file_path[:index] + path_prefix + txt_file_path[index:]

		    #=====[ Read lines from stat file ]=====
		    with open(txt_file_path) as to_read:
			lines = to_read.readlines()

		
		    cur_num_people = 0
		    cur_num_misses = 0
		    cur_num_FP = 0

		    #=====[ Iterate through lines and pull appropriate stat from each ]=====
		    for line in lines:

			try:

			    #=====[ Calculates number of people in frame by looking at the numger of ground truth bounding boxes
			    #=====[ Takes total numbers in ground truth line and divides by 4 (4 parameters for each bounding box)
			    if 'gtruths:' in line:
				gtruths = [x for x in re.split(' |;', line) if x]
				print gtruths
				print len(gtruths)
				print len(gtruths)/4
				cur_num_people = len(gtruths)/4
			    
			    if 'n_FP:' in line:
				cur_num_FP = int(line.split(':')[1].strip())

			    if 'n_misses:' in line:
				cur_num_misses = int(line.split(':')[1].strip())
				num_processed += 1
				added = True

			except Exception as e:

			    #=====[ Decrement our counter if we had wrongfully incremented it after encountering 'num_misses' in a line ]=====
			    num_processed -= 1 if added else 0
			    cur_num_people = 0
		 	    cur_num_FP = 0
			    cur_num_misses = 0
			    print e
			    print '-----> Could not find stats in image file: ', idx + 1
			    continue

		    num_people += cur_num_people
		    num_misses += cur_num_misses
		    num_FP += cur_num_FP

		#=====[ Try Catch around each image in case one is corrupted, non-existant, etc ]=====
	    except Exception as e:
	    	print e
		print '-----> Could not aggregate stats for image: ', idx + 1
		continue

	    print 'TOTAL NUMBER OF PEOPLE', num_people
	return (num_people, num_people - num_misses, num_misses, num_FP, num_processed)










