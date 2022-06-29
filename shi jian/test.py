import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import imageio
#import seaborn as sns
## 输出图显示中文
from matplotlib.font_manager import FontProperties
fonts = FontProperties(fname = "/System/Library/Fonts/Apple Symbols.ttf")
from PIL import Image
# import re
# import string
# import copy
# import time
#from sklearn.metrics import accuracy_score,confusion_matrix
 
#from nltk.corpus import stopwords
import torch
from torch import nn
import torch.nn.functional as F
#import torch.optim as optim
import torch.utils.data as Data
from jieba.analyse import *
# import jieba
import synonyms
import gradio as gr

#logging.basicConfig(level=logging.DEBUG)
#jieba.setLogLevel(logging.INFO)

def Wordcloud(word_dict):  #词云图
    myfont=r'/System/Library/Fonts/meituantype-Bold.TTF' #字体路径（黑体）
    text = word_dict  
    background = r'pict.jpg'  # 背景图片
    word_cloud = WordCloud(font_path=myfont,mask = imageio.imread(background),background_color='white').fit_words(text)
    
    plt.imshow(word_cloud)
    plt.axis("off")  #坐标轴关闭
    plt.show
    
    word_cloud.to_file('词云图.png')  #导出图片
    return word_cloud

def similar_cal(sentence):
  summ=0.0
  sum_w=0.0
  for sen in sentence:
    sum_w=sum_w+sen[1]
  for w in sentence:
    W=w[1]/sum_w
    summ=summ+W*synonyms.compare('食品安全',w[0],seg=True)
  return summ

def write_food(content):
  Note=open('food_safe.txt',mode='a')
  Note.write(content+'\n') 
  Note.close()

def text_deal(textpath):
  with open(textpath,encoding='utf-8') as file:
    texts=file.read()
    texts=texts.split('\n')
  for line in texts:
    text=line
    a=extract_tags(text, topK = 6, withWeight = True, allowPOS = ())
    flag=0.3
    res=similar_cal(a)
    print(res)
    if res<flag:
      print("非食品安全新闻")
    else:
      print("食品安全新闻")
      write_food(line)
      
  with open('food_safe.txt',encoding='utf-8') as file:
    final_news=file.read()
    #final_news=final_news.split('\n')
    a=extract_tags(final_news, topK = 11, withWeight = True, allowPOS = ())
    word_dict = {} #转化为字典形式以便做词云图
    for i in a:
      word_dict[i[0]]=i[1]         
    print(word_dict)
    wcd=Wordcloud(word_dict)
    wcd.to_file('ciyuntu.jpg')
    picw=np.array(Image.open('ciyuntu.jpg'))
  return texts,final_news,picw     #,wcd
# texts_path='data/news.txt'
# text_deal(texts_path)

gr.Interface(fn=text_deal, inputs='text', outputs=['text','text','image'], capture_session=True).launch(share=True)


