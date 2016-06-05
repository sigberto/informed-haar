from opt_pipeline import Pipeline
import pickle

data_file = 'top_100_ft_vectors.p'
model_file = 'top_ft_classifier_1000'

pipe = Pipeline()
#X, Y = pipe.extract_features(file_name=data_file)
#data = pickle.load(open(data_file))

#X = data['input']
#Y = data['labels']

#pipe.train(X, Y, num_estimators=200, max_depth=3, model_name=model_file)
pipe.detect('top_1000_',offset=0, scaling_factor=1.2, scaling_iters=3, nms=0.5,clf=model_file)
