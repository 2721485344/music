#coding:utf-8

from pydub.audio_segment import AudioSegment#pydub是python中用户处理音频文件的一个库
from scipy.io import wavfile
from python_speech_features.base import mfcc #傅里叶变换+梅尔倒谱
import pandas as pd
import numpy as np
import sys


#mfcc 包含了两个步骤，一个是傅里叶变换，一个是梅尔倒谱系数
song = AudioSegment.from_file('./data/灰姑娘.mp3', format = 'mp3')#读入歌曲
# song_split = song[-30*1000:]#切分歌曲
song.export('./data/灰姑娘.wav', format= 'wav')#MP3到wav的转换
rate, data = wavfile.read('./data/灰姑娘.wav')#每秒播放速度及数据
mf_feat = mfcc(data, rate, numcep = 13, nfft = 2048)#傅里叶变换速度每秒多少帧
#  numcep = 13 越大越慢
# 108键， 小于1/4 欢快，大于1/4悲伤

print(mf_feat)
print(mf_feat.shape)
sys.exit(0)
# df = pd.DataFrame(mf_feat)
# df.to_csv('./mfFeat.csv')
# print(mf_feat)
# print(mf_feat.shape)
mm = np.mean(mf_feat, axis = 0)#隐含了时域上的相关性
mf = np.transpose(mf_feat)
mc = np.cov(mf) #原mf_feat矩阵列的协方差矩阵
# print(mc)
result = mm
# 举例说明diag矩阵x = np.arange(10, 19).reshape((3, 3))
for k in range(len(mm)):
    result = np.append(result, np.diag(mc, k))#对角线
#     print(result)
print(result)