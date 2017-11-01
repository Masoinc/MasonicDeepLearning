import numpy as np


# Min-Max 标准化
def mnormalize(data):
    if (np.max(data) - np.min(data)) != 0:
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    else:
        return 0


# Z-Score 标准化
def znormalize(data):
    return (data - np.mean(data)) / np.std(data)

# 反标准化
def unmnormalize(data, srcdata):
    return data * np.max(srcdata)
