import autogluon.core as ag
from autogluon.vision import ImageDataset,ImagePredictor
import pandas as pd
import os
csv_file = ag.utils.download('https://autogluon.s3-us-west-2.amazonaws.com/datasets/petfinder_example.csv')
df = pd.read_csv(csv_file)
df.head()

df = ImageDataset.from_csv(csv_file, root='/home/mzy')
df.head()

train_data, _, test_data = ImageDataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip', train='train', test='test')
print('train #', len(train_data), 'test #', len(test_data))
train_data.head()

# use the train from shopee-iet as new root
root = os.path.join(os.path.dirname(train_data.iloc[0]['image']), '..')
all_data = ImageDataset.from_folder(root)
all_data.head()

# you can manually split the dataset or use `random_split`
train, val, test = all_data.random_split(val_size=0.1, test_size=0.1)
print('train #:', len(train), 'test #:', len(test))

print(train_data)

predictor = ImagePredictor()
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(train_data, hyperparameters={'epochs': 2})  # you can trust the default config, we reduce the # epoch to save some build time

image_path = test_data.iloc[1]['image']
result = predictor.predict(image_path)
print(result)

proba = predictor.predict_proba(image_path)
print(proba)

