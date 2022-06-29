import pandas as pd
import gensim.models
from gensim.models import word2vec
path = 'yf_amazon/'
products = pd.read_csv(path + 'products.csv')
with open('sentence.txt','a+') as f:    # 现在jupyter新建一个txt空文档
    for line in products.values:
        f.write((str(line[0])+'\t'+str(line[1])+'\t'+str(line[2])+'\n'))
        #展示五列 ，如果不能用str，可以先cast as varchar(20)
sentencess = word2vec.Text8Corpus('sentence.txt')
model = gensim.models.Word2Vec(sentences=sentencess)
#model.save('wordpro.model')
print(model.similarity('食品', '蘑菇') )