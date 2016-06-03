from pipeline import Pipeline
import pickle

top_templates = pickle.load(open('notebooks/top_templates.p','rb'))

pipe = Pipeline(top_templates)
X, Y = pipe.extract_features(file_name='top_training_info.p')

pipe.train(X,Y, num_estimators=200, max_depth=2, model_name='top_ft_classifier')