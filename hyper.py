from parlai.scripts.train_model import TrainModel

# note that if you want to see model-specific arguments, you must specify a model name
print(TrainModel.help(model='seq2seq'))