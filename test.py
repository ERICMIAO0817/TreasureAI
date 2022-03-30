# from autogluon.vision import ImagePredictor, ImageDataset
# import pandas as pd
# import matplotlib.pyplot as plt # plt 用于显示图片
# import matplotlib.image as mpimg # mpimg 用于读取图片
# predictor=ImagePredictor.load(path='/home/mzy/PycharmProjects/pythonProject/multi-classification/.trial_5/best_checkpoint.pkl')
# res=predictor.predict('/home/mzy/PycharmProjects/pythonProject/garbe/harm/1..jpg')
# print(res)
# print(res.columns)
# data = pd.DataFrame(res, columns=['id'])
#
# data.loc[res['id'] == 0]='有害垃圾'
# data.loc[res['id'] == 1]='厨余垃圾'
# data.loc[res['id'] == 2]='可回收垃圾'
# data.loc[res['id'] == 3]='其他垃圾'
#
# lena = mpimg.imread('/home/mzy/Downloads/10.jpg') # 读取和代码处于同一目录下的 lena.png
# # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
# lena.shape #(512, 512, 3)
# plt.imshow(lena)
# plt.show()
# print(data.loc[[0]])
import autogluon.core as ag
from autogluon.vision import ImagePredictor, ImageDataset
import pandas

predictor = ImagePredictor().load('.trial_5/best_checkpoint.pkl')
# print(predictor.predict('/media/csu001/存储/autogluon/trash/大骨头/fimg_839.jpg'))

res = predictor.predict('/home/mzy/Downloads/5.jpg')
# print(res)
id_to_class = {
    0: '一次性餐具',
    1: '传单',
    2: '充电宝',
    3: '剩菜剩饭',
    4: '包',
    5: '化妆品瓶',
    6: '卫生纸',
    7: '塑料玩具',
    8: '塑料碗盆',
    9: '塑料衣架',
    10: '大骨头',
    11: '尿片',
    12: '干电池',
    13: '废弃水银温度计',
    14: '废旧灯管灯泡',
    15: '快递纸袋',
    16: '报纸',
    17: '插头电线',
    18: '旧书',
    19: '旧衣服',
    20: '易拉罐',
    21: '杀虫剂容器',
    22: '杂志',
    23: '枕头',
    24: '果壳瓜皮',
    25: '残枝落叶',
    26: '毛绒玩具',
    27: '水果果皮',
    28: '水果果肉',
    29: '污损塑料',
    30: '泡沫塑料',
    31: '洗发水瓶',
    32: '烟蒂',
    33: '牙签',
    34: '牛奶盒等利乐包装',
    35: '玻璃',
    36: '玻璃瓶罐',
    37: '电池',
    38: '皮鞋',
    39: '砧板',
    40: '破碎花盆及碟碗',
    41: '竹筷',
    42: '纸杯',
    43: '纸板箱',
    44: '茶叶渣',
    45: '菜梗菜叶',
    46: '落叶',
    47: '蛋壳',
    48: '西餐糕点',
    49: '调料瓶',
    50: '贝壳',
    51: '软膏',
    52: '过期药物',
    53: '酒瓶',
    54: '金属食品罐',
    55: '锅',
    56: '除草剂容器',
    57: '食用油桶',
    58: '饮料瓶',
    59: '鱼骨'
}
res['prediction'] = res['id'].map(id_to_class)
print(res)