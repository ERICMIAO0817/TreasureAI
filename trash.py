from autogluon.vision import ImagePredictor, ImageDataset
dataset = ImageDataset.from_folder('/home/mzy/PycharmProjects/pythonProject/garbe/')
print(dataset)
time_limit = 10 * 60 # 10mins
predictor = ImagePredictor()
predictor.fit(dataset, time_limit=None,presets='medium_quality_faster_train',hyperparameters={'batch_size':16})

predictor.predict('/home/mzy/PycharmProjects/pythonProject/garbe/harm/3..jpg')