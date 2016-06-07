from opt_pipeline import Pipeline
import pickle

data_file = 'top_250_ft_vectors.p'
model_file = 'top_ft_classifier'

pipe = Pipeline()
#X, Y = pipe.extract_features(num_ft=250,file_name=data_file)
#data = pickle.load(open(data_file))

#X = data['input']
#Y = data['labels']

#pipe.train(X, Y, num_estimators=500, max_depth=2, model_name=model_file)
#pipe.detect(output_file_prefix='final_scale_nms_7_', num_ft=100, offset=0, scaling_factor=1.33, scaling_iters=6, nms=0.7,clf=model_file, old_detector=False)

pipe.get_stats(output_file_prefix='top_250_500_', num_images=200)
