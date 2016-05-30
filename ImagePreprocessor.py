import cv2
import numpy as np
import os
from math import trunc
from random import randint

class ImagePreprocessor:

    def __init__(self, source_train_dir=None, source_test_dir=None, dest_train_dir=None, dest_test_dir=None):
        if source_train_dir is None:
            self.source_train_dir = 'INRIAPerson/train_64x128_H96/'
        else:
            self.source_train_dir = source_train_dir

        if source_test_dir is None:
            self.source_test_dir = 'INRIAPerson/test_64x128_H96/'
        else:
            self.source_test_dir = source_test_dir

        if dest_train_dir is None:
            self.dest_train_dir = 'INRIAPerson/train_us/'
        else:
            self.dest_train_dir = dest_train_dir

        if dest_test_dir is None:
            self.dest_test_dir = 'INRIAPerson/test_us/'
        else:
            self.dest_test_dir = dest_test_dir

        self.base_dir = 'INRIAPerson/'
        self.train_pos_size = (160, 96)
        self.test_pos_size = (134, 70)
        # negative images do not have a determined size, so we don't account for that
        self.result_size = (120, 60, 3)
        self.neg_crops_per_neg_image = 4

    def crop_and_save_pos(self, img_paths, is_train):
        if is_train:
            row_offset = trunc((self.train_pos_size[0] - self.result_size[0])/2)
            col_offset = trunc((self.train_pos_size[1] - self.result_size[1])/2)
        else:
            row_offset = trunc((self.test_pos_size[0] - self.result_size[0])/2)
            col_offset = trunc((self.test_pos_size[1] - self.result_size[1])/2)

        for path in img_paths:
            path = self.base_dir + path
            img = cv2.imread(path)
            img = img[row_offset:row_offset + self.result_size[0], col_offset:col_offset + self.result_size[1], :]
            assert (img.shape == self.result_size), 'Image resizing failed: expected {}, got {}'.format(str(self.result_size), str(img.shape))
            if is_train:
                dest = path.replace(self.source_train_dir, self.dest_train_dir)
            else:
                dest = path.replace(self.source_test_dir, self.dest_test_dir)
            cv2.imwrite(dest, img)

    def crop_and_save_neg(self, img_paths, is_train):
        for path in img_paths:
            path = self.base_dir + path
            img = cv2.imread(path)
            for i in xrange(self.neg_crops_per_neg_image):
                size = img.shape
                rand_row = randint(0, size[0] - self.result_size[0] - 1)
                rand_col = randint(0, size[1] - self.result_size[1] - 1)
                sub_im = img[rand_row:rand_row + self.result_size[0], rand_col:rand_col + self.result_size[1], :]
                assert (sub_im.shape == self.result_size), 'Image resizing failed: expected {}, got {}'.format(str(self.result_size), str(sub_im.shape))
                if is_train:
                    dest = path.replace('train_64x128_H96', 'train_us')
                else:
                    dest = path.replace('test_64x128_H96', 'test_us')
                dest = dest.replace('.', '_{}.'.format(str(i)))
                cv2.imwrite(dest, sub_im)

    def make_result_file_list(self, path, subdir, list_name):
        files = []
        for (_, _, filenames) in os.walk(os.path.join(path, subdir)):
            files.extend(filenames)
            break
        files = [file + '\n' for file in files]
        with open(os.path.join(path, list_name), 'wb') as f:
            f.writelines(files)

    def preprocess_images(self):
        with open(self.source_train_dir + 'pos.lst') as f:
            train_pos_list = f.readlines()
            train_pos_list = [x.strip() for x in train_pos_list]
            train_pos_list = [x.replace('train', 'train_64x128_H96') for x in train_pos_list]
        with open(self.source_train_dir + 'neg.lst') as f:
            train_neg_list = f.readlines()
            train_neg_list = [x.strip() for x in train_neg_list]
            train_neg_list = [x.replace('train', 'train_64x128_H96') for x in train_neg_list]
        with open(self.source_test_dir + 'pos.lst') as f:
            test_pos_list = f.readlines()
            test_pos_list = [x.strip() for x in test_pos_list]
            test_pos_list = [x.replace('test', 'test_64x128_H96') for x in test_pos_list]
        with open(self.source_test_dir + 'neg.lst') as f:
            test_neg_list = f.readlines()
            test_neg_list = [x.strip() for x in test_neg_list]
            test_neg_list = [x.replace('test', 'test_64x128_H96') for x in test_neg_list]

        self.crop_and_save_pos(train_pos_list, is_train=True)
        self.crop_and_save_pos(test_pos_list, is_train=False)
        self.crop_and_save_neg(train_neg_list, is_train=True)
        self.crop_and_save_neg(test_neg_list, is_train=False)

        self.make_result_file_list(self.dest_train_dir, 'pos', 'pos.lst')
        self.make_result_file_list(self.dest_test_dir, 'pos', 'pos.lst')
        self.make_result_file_list(self.dest_train_dir, 'neg', 'neg.lst')
        self.make_result_file_list(self.dest_test_dir, 'neg', 'neg.lst')

