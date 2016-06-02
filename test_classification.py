from classifier import Classifier

clf = Classifier()


#=====[ Get positive and negative image paths ]=====
raw_img_files = open('../test_us/pos.lst','rb')
pos_img_paths = ['../test_ust/pos/'+ path.strip() for path in raw_img_files.readlines()]
raw_img_files.close()

raw_img_files = open('../test_us/neg.lst','rb')
neg_img_paths = ['../test_ust/neg/'+ path.strip() for path in raw_img_files.readlines()]
raw_img_files.close()

#=====[ Get num of positive/negative images ]=====
n_pos = len(pos_img_paths)
n_neg = len(neg_img_paths)

#=====[ Get positive image accuracy ]=====
Y_pos = np.ones((len(pos_img_paths,)))
accuracy = clf.test(img_paths)*n_pos/(n_pos + n_neg)

#=====[ Get negative image accuracy ]=====
Y_neg = np.zeros((len(neg_img_paths),))
accuracy += clf.test(img_paths)*n_neg/(n_post + n_neg)

print accuracy 
print 'Accuracy on %d images: %f ' (n_pos+n_neg, accuracy)
