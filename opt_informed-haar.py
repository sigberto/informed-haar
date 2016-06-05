from opt_pipeline import Pipeline

# data_file = 'top_100_ft_vectors.p'
model_file = 'top_ft_classifier_1000'

pipe = Pipeline()
# X, Y = pipe.extract_features(file_name=data_file)
#pipe.train(X, Y, num_estimators=2000, max_depth=2, model_name=model_file)
pipe.detect(clf=model_file)
