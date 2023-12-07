#    Copyright 2020 Arkadip Bhattacharya

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# 这段代码定义了一个用于风速数据的自定义数据集类。
# 这个类接收了一个pandas数据框架和一个可选的转换函数。
# 它从数据框架中删除了 "时间 "列，并将 "风速 "列设置为标签集。
# 其余的列被设置为特征集。
# __len__方法返回特征集的长度。
# __getitem__方法返回一个给定索引的特征和标签的元组。
# 如果提供了一个转换函数，它在返回样本之前会对其进行转换。
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class WindPowerDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        dataframe1 = dataframe.copy()
        self.transform = transform
        
        if '时间' in dataframe1:
            dataframe1.pop('时间')
        
        if '实际发电功率' in dataframe1:
            self.labelset = dataframe1.pop('实际发电功率')
            
        self.featureset = dataframe1
        
    def __len__(self):
        return len(self.featureset)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print(idx)
        
        # Get the label and features for the given index
        label = np.array([self.labelset.iloc[idx]])
        features = self.featureset.iloc[idx].to_numpy()
        
        sample =(features, label)
                
        # Apply the transform function if provided
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
    
# 这段代码定义了一个用于风速时间序列数据的自定义数据集类。
# 这个类接收了一个pandas数据框架，一个窗口大小和一个可选的转换函数。
# 它从数据框架中删除了 "时间 "列，并将 "风速 "列设置为标签集。
# 其余的列被设置为特征集。
# __len__方法返回特征集的长度减去窗口大小再减去1。
# __getitem__方法返回一个给定索引的特征和标签的元组。
# 如果提供了一个转换函数，它在返回样本之前会对其进行转换。 


class WindPowerTimeSeriesDataset(Dataset):
    def __init__(self, dataframe, window_size=6, transform=None):
        dataframeC = dataframe.copy()
        self.transform = transform
        self.window_size = window_size
        
        if '时间' in dataframeC:
            dataframeC.pop('时间')
        
        if '实际发电功率' in dataframeC:
            self.labelset = dataframeC['实际发电功率']
            
        self.featureset = dataframeC
        #print(self.featureset.shape)
         

    def __len__(self):
          # 返回特征集的长度减去窗口大小再减去1
        return math.floor(len(self.featureset) - self.window_size) - 1
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print(idx)
        #print("__getitem__")
        # 获取给定索引的标签和特征
        label = np.array([self.labelset.iloc[idx+self.window_size]])
        features = self.featureset.iloc[idx:idx+self.window_size].to_numpy()
        #print("features",features)
        #print("label",label)
        # 创建一个特征和标签的元组
        sample = (features, label)
        #print(label.shape)
        #print(features.shape)
        # 如果提供了一个转换函数，它在返回样本之前会对其进行转换。
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    
    

class ComposeTransform(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> ComposeTransform([
        >>>     ToTensor(),
        >>> ])
        
        "将几个变换组合在一起。

    参数：
        transforms (列表中的``Transform``对象)：要组合的变换的列表。

    例子：
        >>> ComposeTransform([
        >>> ToTensor()、
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            img = t(data)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample
        return (torch.from_numpy(data),torch.from_numpy(label))