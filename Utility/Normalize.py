import numpy as np


# Min-Max 标准化
def mnormalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Z-Score 标准化
def znormalize(data):
    return (data - np.mean(data)) / np.std(data)

# 反标准化
def unmnormalize(data, srcdata):
    return data * np.max(srcdata)
